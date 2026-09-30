#!/usr/bin/env python3
"""gen_training_pairs.py - generate the NRR training pairs we own outright.

Why generated and not downloaded: weights are only half of a model's licensing surface - the data
has one too, and a public SR dataset brings its own terms. Everything here is produced from a seed,
so the pairs, the ground-truth depth and the ground-truth motion vectors are ours, and a dataset is
reproducible from (this revision, the seed).

What a pair is, and why each part is ground truth:
    input   low-resolution degraded render (noise added), the frame a runtime receives
    target  the same scene at 2x with supersampling - the image the model should produce
    depth   per-pixel ray distance from the intersection test (not estimated)
    motion  per-pixel screen-space displacement of the shaded point between t and t+dt, computed from
            the scene's velocities (not estimated). Translation only: a rotating body would need the
            full transform, and saying so is cheaper than pretending.

The gates run on every pair, and they exist because of a specific failure this project already had:
the shipped fixture is an untrained identity function, so a generator that accidentally produced
input == target would train another identity and look like progress. The gates make that impossible -
after a naive bilinear 2x upscale the input must still differ from the target by a stated margin, and
depth and motion must be non-constant (a constant conditioning input trains a model that ignores
conditioning, which is the gap the roadmap records).

Usage:
    python tools/gen_training_pairs.py --out models/training-data/v1 --count 8 --size 64
    python tools/gen_training_pairs.py --self-test     # prove the gate rejects an identical pair

Requires: pip install numpy
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# A pair whose input survives a naive 2x upscale to within this much of the target has nothing for a
# model to learn. 0.01 in [0,1] colour units is about one 8-bit level of mean error.
MIN_IDENTITY_MARGIN = 0.01


class Scene(object):
    """A deterministic procedural scene: spheres over a gradient sky, one of them moving so motion
    vectors mean something. `style` varies palette and layout, so a held-out style is a real
    held-out case rather than the same scene with a different seed."""

    def __init__(self, seed, style=0):
        self.seed = int(seed)
        self.style = int(style)
        rng = np.random.default_rng(self.seed * 7919 + self.style * 104729)

        count = 3 + (self.style % 3)
        self.centers = np.zeros((count, 3), dtype=np.float64)
        self.radii = np.zeros(count, dtype=np.float64)
        self.albedo = np.zeros((count, 3), dtype=np.float64)
        self.velocity = np.zeros((count, 3), dtype=np.float64)
        for i in range(count):
            self.centers[i] = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.9, 0.9),
                                        rng.uniform(3.0, 6.0)])
            self.radii[i] = rng.uniform(0.5, 1.1)
            self.albedo[i] = rng.uniform(0.2, 0.9, size=3)
        # Exactly one sphere moves: motion stays non-constant and its cause stays obvious. It is placed
        # in front of the camera on purpose - a mover that happens to fall outside the frustum leaves
        # motion identically zero, and an earlier revision of this generator did exactly that. The
        # motion gate caught it, which is why the gate exists.
        self.centers[0] = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.35, 0.35),
                                    rng.uniform(3.6, 4.6)])
        self.radii[0] = rng.uniform(0.6, 0.9)
        self.albedo[0] = rng.uniform(0.35, 0.8, size=3)
        self.velocity[0] = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.2, 0.2), 0.0])
        self.light = np.array([0.4, 0.7, -0.5])
        self.light = self.light / np.linalg.norm(self.light)
        self.sky_top = np.array([0.35, 0.45, 0.75]) * (1.0 + 0.15 * self.style)
        self.sky_bottom = np.array([0.75, 0.70, 0.60])


def _trace(scene, height, width, time_offset, supersample):
    """Renders one image at one time: colour, depth, and which sphere each pixel hit (so motion can
    come from the geometry rather than from the pixels)."""
    steps = supersample if supersample > 1 else 1
    color = np.zeros((height, width, 3))
    depth = np.zeros((height, width))
    ids = np.full((height, width), -1, dtype=np.int32)

    for sy in range(steps):
        for sx in range(steps):
            directions = _ray_directions(height, width)
            if steps > 1:
                directions[..., 0] += ((sx + 0.5) / steps - 0.5) * (2.0 / width)
                directions[..., 1] -= ((sy + 0.5) / steps - 0.5) * (2.0 / height)
            directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

            best_t = np.full((height, width), np.inf)
            hit_id = np.full((height, width), -1, dtype=np.int32)
            hit_point = np.zeros((height, width, 3))
            for index in range(scene.centers.shape[0]):
                center = scene.centers[index] + scene.velocity[index] * time_offset
                b = np.sum(directions * (-center), axis=-1)
                c = float(np.dot(center, center)) - scene.radii[index] ** 2
                disc = b * b - c
                hit = disc > 0.0
                if not np.any(hit):
                    continue
                t = -b - np.sqrt(np.where(hit, disc, 0.0))
                closer = hit & (t > 1e-4) & (t < best_t)
                best_t = np.where(closer, t, best_t)
                hit_id = np.where(closer, index, hit_id)
                hit_point = np.where(closer[..., None], directions * t[..., None], hit_point)

            sky = np.linspace(0.0, 1.0, height)[:, None, None]
            frame = scene.sky_top * (1.0 - sky) + scene.sky_bottom * sky
            for index in range(scene.centers.shape[0]):
                mask = hit_id == index
                if not np.any(mask):
                    continue
                center = scene.centers[index] + scene.velocity[index] * time_offset
                normal = hit_point - center
                normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
                lambert = np.clip(np.sum(normal * scene.light, axis=-1), 0.0, 1.0)
                shade = (0.25 + 0.75 * lambert)[..., None] * scene.albedo[index]
                frame = np.where(mask[..., None], shade, frame)

            color += frame
            depth += np.where(np.isfinite(best_t), best_t, 0.0)
            ids = np.where(hit_id >= 0, hit_id, ids)

    samples = float(steps * steps)
    return color / samples, depth / samples, ids


def render_pair(scene, size):
    """A degraded low-resolution input, a supersampled high-resolution target, and the depth and
    motion that belong to the input's grid."""
    target_color, _unused_depth, _unused_ids = _trace(scene, size * 2, size * 2, 0.0, 2)
    input_color, input_depth, input_ids = _trace(scene, size, size, 0.0, 1)

    rng = np.random.default_rng(scene.seed * 31 + 17)
    input_color = np.clip(input_color + rng.normal(0.0, 0.02, size=input_color.shape), 0.0, 1.0)

    # Motion: where each shaded point lands 50 ms later, in input-pixel units. Only the moving sphere
    # contributes, because neither the camera nor the other spheres move.
    directions = _ray_directions(size, size)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    safe_depth = np.where(input_depth > 0.0, input_depth, 1.0)
    points = directions * safe_depth[..., None]
    points_later = points + scene.velocity[0] * 0.05 * (input_ids == 0)[..., None]
    scale = float(size) / 2.0
    motion = np.zeros((size, size, 2))
    motion[..., 0] = (points_later[..., 0] / points_later[..., 2] -
                      points[..., 0] / points[..., 2]) * scale
    motion[..., 1] = -(points_later[..., 1] / points_later[..., 2] -
                       points[..., 1] / points[..., 2]) * scale

    return {"input": input_color.astype(np.float32),
            "target": np.clip(target_color, 0.0, 1.0).astype(np.float32),
            "depth": safe_depth.astype(np.float32), "motion": motion.astype(np.float32)}


def _ray_directions(height, width):
    """Pinhole camera down +z, not moving: then every motion vector in the data has exactly one known
    cause (the moving sphere)."""
    aspect = float(width) / float(height)
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0
    gx, gy = np.meshgrid(xs * aspect, ys)
    return np.stack([gx, gy, np.ones_like(gx)], axis=-1)


def _upscale2x(image):
    """Bilinear 2x upscale: the naive baseline a model has to beat, used to measure whether the input
    differs from the target at all."""
    height, width = image.shape[:2]
    ys = (np.arange(height * 2) + 0.5) / 2.0 - 0.5
    xs = (np.arange(width * 2) + 0.5) / 2.0 - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, height - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    fy = np.clip(ys - y0, 0.0, 1.0)[:, None, None]
    fx = np.clip(xs - x0, 0.0, 1.0)[None, :, None]
    top = image[y0][:, x0] * (1.0 - fx) + image[y0][:, x1] * fx
    bottom = image[y1][:, x0] * (1.0 - fx) + image[y1][:, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def gate_pair(name, pair):
    """Raises SystemExit when a pair would train an identity function or a conditioning-blind model;
    returns the measured margin so the run can print it (a gate that does not say what it measured is
    a gate nobody can tune)."""
    margin = float(np.mean(np.abs(_upscale2x(pair["input"]) - pair["target"])))
    if margin <= MIN_IDENTITY_MARGIN:
        raise SystemExit(
            "%s: after a bilinear 2x upscale the input is within %.6f of the target (needs > %.4f) - "
            "this pair has nothing to learn and would train another identity function"
            % (name, margin, MIN_IDENTITY_MARGIN))
    for field in ("depth", "motion"):
        spread = float(np.std(pair[field]))
        if spread <= 1e-6:
            raise SystemExit("%s: %s is constant (std %.9f) - a model trained on this would ignore "
                             "its conditioning" % (name, field, spread))
    return margin


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(out_dir, count, size, seed, val_every, style):
    os.makedirs(out_dir, exist_ok=True)
    entries, margins = [], []
    for index in range(count):
        pair_seed = seed + index * 101
        pair = render_pair(Scene(pair_seed, style), size)
        split = "val" if (val_every > 0 and index % val_every == 0) else "train"
        name = "%s_%03d.npz" % (split, index)
        margin = gate_pair("pair %d (seed %d)" % (index, pair_seed), pair)
        margins.append(margin)
        path = os.path.join(out_dir, name)
        np.savez_compressed(path, **pair)
        entries.append({"file": name, "split": split, "seed": pair_seed, "style": style,
                        "size": size, "identity_margin": margin, "sha256": _hash(path)})
        print("  %-14s split=%-5s seed=%-11d margin=%.4f depth_std=%.4f motion_std=%.4f"
              % (name, split, pair_seed, margin, float(np.std(pair["depth"])),
                 float(np.std(pair["motion"]))))

    manifest = {"generator": os.path.basename(__file__), "count": count, "size": size, "seed": seed,
                "style": style, "min_identity_margin": MIN_IDENTITY_MARGIN,
                "margins": {"min": min(margins), "mean": float(np.mean(margins))}, "pairs": entries}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("  manifest: %s (%d pairs, min margin %.4f, mean %.4f)"
          % (manifest_path, count, manifest["margins"]["min"], manifest["margins"]["mean"]))
    return manifest


def self_test():
    """Proves the gates are not decorative: two degenerate pairs must be rejected. The first has an
    input that a naive upscale reproduces exactly, which is the failure the shipped fixture represents;
    the second has conditioning that cannot be conditioned on, which is the failure a scene without a
    visible mover produces - an earlier revision of this generator produced exactly that, and the gate
    caught it."""
    size = 16
    ramp = np.repeat((np.linspace(0.0, 1.0, size * 2)[:, None] *
                      np.ones((1, size * 2)))[..., None], 3, axis=-1)
    cases = [
        ("identity input", {"input": ramp[::2, ::2].copy(), "target": ramp.copy(),
                            "depth": np.linspace(1.0, 4.0, size)[:, None] * np.ones((1, size)),
                            "motion": np.zeros((size, size, 2))}),
        ("constant conditioning", {"input": np.zeros((size, size, 3)),
                                   "target": np.ones((size * 2, size * 2, 3)),
                                   "depth": np.ones((size, size)),
                                   "motion": np.zeros((size, size, 2))}),
    ]
    failures = 0
    for name, pair in cases:
        try:
            margin = gate_pair(name, pair)
        except SystemExit as rejected:
            print("  %-21s rejected as it must be: %s" % (name, rejected))
        else:
            print("  %-21s NOT REJECTED (margin %.4f) - the gate is decorative" % (name, margin))
            failures += 1
    return 1 if failures else 0


def main(argv):
    parser = argparse.ArgumentParser(description="Generate the NRR training pairs we own outright.")
    parser.add_argument("--out", default="models/training-data/v1")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--val-every", type=int, default=4)
    parser.add_argument("--style", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv[1:])
    if args.self_test:
        return self_test()
    print("generating %d pairs at %dx%d (target %dx%d) into %s"
          % (args.count, args.size, args.size, args.size * 2, args.size * 2, args.out))
    generate(args.out, args.count, args.size, args.seed, args.val_every, args.style)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
