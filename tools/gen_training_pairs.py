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
input == target would train another identity and look like progress. Three things are measured, and
each one is printed and recorded in the manifest rather than assumed:

    identity margin   after a naive bilinear 2x upscale the input must still differ from the target
    detail ratio      high-frequency energy of that upscale over the target's - a super-resolution
                      pair is only useful if the target really carries more detail than the baseline
    conditioning      depth and motion must be non-constant, because a constant conditioning input
                      trains a model that ignores conditioning, which is the gap the roadmap records

The detail ratio exists because the first version of this generator measured only the identity margin,
which sat at 0.011 against a 0.01 gate - it passed, with almost no headroom, because the seeded noise
(mean |noise| ~ 0.016) was larger than the resolution difference it was supposed to be measuring. A
margin that is mostly noise trains a denoiser, not an upscaler. Noise is now 0.005, the noise floor is
reported separately, and the detail ratio is the gate that actually asks whether there is resolution
to recover. Noise can only inflate the input's detail and therefore only make the gate harder to pass,
so the measurement errs against the generator, which is the direction that matters.

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

# The naive upscale may keep at most this fraction of the target's high-frequency energy. At 1.0 the
# target has no more detail than the baseline and the pair is a denoising exercise; 0.9 leaves room for
# the pair to be worth training on without flattering it.
MAX_DETAIL_RATIO = 0.9

# Sensor-ish noise added to the input. 0.005 keeps mean |noise| (~0.004) well under the resolution
# difference, so the pair tests upscaling rather than denoising - see the docstring.
INPUT_NOISE_SIGMA = 0.005



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

            # Ground plane at y = -1.2 with a checker pattern. Analytic, so it needs no texture, and its
            # edges at grazing angles are precisely what a low-resolution render loses - which is the
            # content the identity and detail gates were failing for want of. -2 marks the ground, so it
            # contributes depth and colour but no motion.
            ground_t = np.where(directions[..., 1] < -1e-6, -1.2 / directions[..., 1], np.inf)
            ground_hit = (ground_t > 1e-4) & (ground_t < best_t)
            best_t = np.where(ground_hit, ground_t, best_t)
            hit_id = np.where(ground_hit, -2, hit_id)

            sky = np.linspace(0.0, 1.0, height)[:, None, None]
            frame = scene.sky_top * (1.0 - sky) + scene.sky_bottom * sky
            if np.any(ground_hit):
                # Sky rays never hit the ground: their t is infinite, so floor() of infinity is the nan
                # and the RuntimeWarning that used to print here. Only ground hits are worth shading.
                ground_point = directions * np.where(ground_hit, ground_t, 1.0)[..., None]
                checker = ((np.floor(ground_point[..., 0] * 0.9) +
                            np.floor(ground_point[..., 2] * 0.9)) % 2.0) == 0.0
                frame = np.where(ground_hit[..., None],
                                 np.where(checker[..., None], 0.82, 0.16), frame)

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
    clean_color = input_color.copy()
    input_color = np.clip(input_color + rng.normal(0.0, INPUT_NOISE_SIGMA, size=input_color.shape),
                          0.0, 1.0)

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
            "input_clean": clean_color.astype(np.float32),
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


def _detail(image):
    """High-frequency energy: mean absolute difference from an 8-neighbour 3x3 blur. A pair is only
    useful for super-resolution if the target carries more of this than a naive upscale of the input."""
    pad = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] +
            pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:]) / 8.0
    return float(np.mean(np.abs(image - blur)))


def gate_pair(name, pair):
    """Raises SystemExit when a pair would train an identity function, a denoiser instead of an
    upscaler, or a conditioning-blind model. Returns the measurements, because a gate that does not say
    what it measured is a gate nobody can tune."""
    upscaled = _upscale2x(pair["input"])
    margin = float(np.mean(np.abs(upscaled - pair["target"])))
    noise = float(np.mean(np.abs(pair["input"] - pair["input_clean"])))
    detail_ratio = _detail(upscaled) / max(_detail(pair["target"]), 1e-9)
    measured = {"margin": margin, "noise": noise, "detail_ratio": detail_ratio}

    if margin <= MIN_IDENTITY_MARGIN:
        raise SystemExit(
            "%s: after a bilinear 2x upscale the input is within %.6f of the target (needs > %.4f) - "
            "this pair has nothing to learn and would train another identity function"
            % (name, margin, MIN_IDENTITY_MARGIN))
    if margin <= noise:
        raise SystemExit(
            "%s: the margin (%.6f) is no larger than the noise floor (%.6f), so the difference this "
            "pair offers is noise rather than resolution - training on it would produce a denoiser"
            % (name, margin, noise))
    if detail_ratio > MAX_DETAIL_RATIO:
        raise SystemExit(
            "%s: a naive upscale already keeps %.3f of the target's high-frequency energy (needs "
            "<= %.2f) - the target has too little detail left to recover" % (name, detail_ratio,
                                                                           MAX_DETAIL_RATIO))
    for field in ("depth", "motion"):
        spread = float(np.std(pair[field]))
        if spread <= 1e-6:
            raise SystemExit("%s: %s is constant (std %.9f) - a model trained on this would ignore "
                             "its conditioning" % (name, field, spread))
    return measured



def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(out_dir, count, size, seed, val_every, style):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for index in range(count):
        pair_seed = seed + index * 101
        pair = render_pair(Scene(pair_seed, style), size)
        split = "val" if (val_every > 0 and index % val_every == 0) else "train"
        name = "%s_%03d.npz" % (split, index)
        stats = gate_pair("pair %d (seed %d)" % (index, pair_seed), pair)
        path = os.path.join(out_dir, name)
        np.savez_compressed(path, **pair)
        entries.append({"file": name, "split": split, "seed": pair_seed, "style": style,
                        "size": size, "margin": stats["margin"], "noise": stats["noise"],
                        "detail_ratio": stats["detail_ratio"], "sha256": _hash(path)})
        print("  %-14s %-5s seed=%-11d margin=%.4f noise=%.4f detail=%.3f depth_std=%.3f "
              "motion_std=%.4f"
              % (name, split, pair_seed, stats["margin"], stats["noise"], stats["detail_ratio"],
                 float(np.std(pair["depth"])), float(np.std(pair["motion"]))))

    def worst(key, pick=max):
        return pick(entry[key] for entry in entries)

    manifest = {"generator": os.path.basename(__file__), "count": count, "size": size, "seed": seed,
                "style": style, "min_identity_margin": MIN_IDENTITY_MARGIN,
                "max_detail_ratio": MAX_DETAIL_RATIO, "input_noise_sigma": INPUT_NOISE_SIGMA,
                "summary": {"margin": {"min": worst("margin", min), "max": worst("margin", max)},
                            "noise": {"min": worst("noise", min), "max": worst("noise", max)},
                            "detail_ratio": {"min": worst("detail_ratio", min),
                                             "max": worst("detail_ratio", max)}},
                "pairs": entries}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("  margin %.4f-%.4f, noise %.4f-%.4f, detail ratio %.3f-%.3f (gate needs margin > %.2f "
          "and > noise, detail <= %.2f)"
          % (manifest["summary"]["margin"]["min"], manifest["summary"]["margin"]["max"],
             manifest["summary"]["noise"]["min"], manifest["summary"]["noise"]["max"],
             manifest["summary"]["detail_ratio"]["min"], manifest["summary"]["detail_ratio"]["max"],
             MIN_IDENTITY_MARGIN, MAX_DETAIL_RATIO))
    print("  manifest: %s (%d pairs)" % (manifest_path, count))
    return manifest



def self_test():
    """Proves the gates are not decorative: three degenerate pairs must be rejected, each by the gate it
    is designed to exercise. The first is the failure the shipped fixture represents - an input a naive
    upscale reproduces exactly, so the target carries no more detail than the baseline. The second is
    the failure the first version of this generator had: a difference that is noise rather than
    resolution. The third is a scene with no visible mover, which an earlier revision produced."""
    size = 16
    ramp = np.repeat((np.linspace(0.0, 1.0, size * 2)[:, None] *
                      np.ones((1, size * 2)))[..., None], 3, axis=-1)
    # Every case carries non-constant depth and motion, so a rejection can only come from the gate the
    # case is designed to exercise - otherwise an unrelated failure would look like a pass.
    depth = np.linspace(1.0, 4.0, size)[:, None] * np.ones((1, size))
    motion = np.zeros((size, size, 2))
    motion[..., 0] = np.linspace(-1.0, 1.0, size)[None, :]
    rng = np.random.default_rng(3)
    noisy = np.clip(ramp[::2, ::2] + rng.normal(0.0, 0.05, size=(size, size, 3)), 0.0, 1.0)

    cases = [
        ("identity input", {"input": ramp[::2, ::2].copy(), "input_clean": ramp[::2, ::2].copy(),
                            "target": ramp.copy(), "depth": depth, "motion": motion}),
        ("noise, no resolution", {"input": noisy, "input_clean": ramp[::2, ::2].copy(),
                                  "target": ramp.copy(), "depth": depth, "motion": motion}),
        ("constant conditioning", {"input": np.zeros((size, size, 3)),
                                   "input_clean": np.zeros((size, size, 3)),
                                   "target": np.ones((size * 2, size * 2, 3)),
                                   "depth": np.ones((size, size)),
                                   "motion": np.zeros((size, size, 2))}),
    ]
    failures = 0
    for name, pair in cases:
        try:
            stats = gate_pair(name, pair)
        except SystemExit as rejected:
            print("  %-21s rejected as it must be: %s" % (name, rejected))
        else:
            print("  %-21s NOT REJECTED (margin %.4f noise %.4f detail %.3f) - the gate is decorative"
                  % (name, stats["margin"], stats["noise"], stats["detail_ratio"]))
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
