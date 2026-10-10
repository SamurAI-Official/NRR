#!/usr/bin/env python3
"""What the resolution token is worth to the model, and what it does outside the range it was trained on.

`upscale_msreal_scale.onnx` is the released scale-agnostic model, and its only description of which tier it is
looking at is the token: log2(input_width / 128), one channel, constant. tools/compare_upscalers.py derives that
value from the image itself and never lets a caller pass one - the right instrument for "how good is this model
at the tiers it was trained on", and the wrong one for the two questions a game at another resolution raises:

  1. **Is the token load-bearing?** If the output is identical at 0.0 and 1.0 the model is not using it and the
     tier is being ignored; if it differs, the value the runtime derives matters, and this measures how much.
  2. **What happens past the trained tiers?** The dataset behind this model holds two: 128 -> 256 (token 0.0) and
     256 -> 512 (token 1.0). A 1080p frame submitted as 960x540 asks for 2.907, and the model never trained
     there. The token is a number, so the graph will run; whether the result is worth having can only be
     answered by scoring it.

The frames are the ones the repository already captured, so the tiers are real: models/training-data/godot-mid is
256 -> 512, and models/training-data/godot-hi is 512 -> 1024 with a captured reference - two octaves past
anything this model was trained on, scored with the token its own width produces. Bilinear is scored beside every
token row because "above the trainer's own baseline" is the property that decides whether the neural pass is
worth its latency at all.

Usage:

  python tools/scale_token_sweep.py --data models/training-data/godot-mid --limit 12
  python tools/scale_token_sweep.py --data models/training-data/godot-hi  --limit 12 \\
      --out work/parity/token-sweep-hi.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as em          # noqa: E402  (the harness's loader, session and metrics)
import quality_metrics as qm         # noqa: E402
from compare_upscalers import build_feed_from_image, pil_upscale  # noqa: E402
from PIL import Image                # noqa: E402

# The token each tier's width produces, plus the two a caller might wonder about: 0.5849 is a 192 px frame's
# value, between the trained tiers, and 2.9069 is a 960 px frame's, the 1080p submission this sweep exists for.
DEFAULT_TOKENS = "0,0.5849625,1,2,2.9068906"


def parse_tokens(text):
    return [float(part) for part in text.split(",") if part.strip()]


def scenes_for(data_dir, split, limit):
    """Frames grouped by scene: the manifest when the dataset has one, the files themselves when it does not.

    A packed dataset here always carries a manifest, but a capture directory that was never packed still holds
    the pairs - models/training-data/godot-hi is one: 512 -> 1024 val pairs, no manifest - and the question this
    script answers does not depend on the packing. Enumerating the files is the fallback, and the file name is
    the scene, so a caller listing them sees which frame each row came from.
    """
    if os.path.exists(os.path.join(data_dir, "manifest.json")):
        return em.group_sequences(em.load_manifest(data_dir), split, limit)
    prefix = "%s_" % split
    names = sorted(name for name in os.listdir(data_dir)
                   if name.startswith(prefix) and name.endswith(".npz"))
    if limit:
        names = names[:limit]
    if not names:
        return {}
    return {os.path.splitext(name)[0]: [{"file": name, "scene": os.path.splitext(name)[0], "frame": index}]
            for index, name in enumerate(names)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="models/training-data/godot-hi",
                        help="a packed dataset directory with a manifest.json")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0, help="cap frames per scene (0 = all)")
    parser.add_argument("--model", default="models/phase4/upscale_msreal_scale.onnx")
    parser.add_argument("--provider", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--out", default="", help="where to write the JSON report")
    args = parser.parse_args(argv)

    tokens = parse_tokens(args.tokens)
    scenes = scenes_for(args.data, args.split, args.limit)
    if not scenes:
        raise SystemExit("no %s frames in %s" % (args.split, args.data))

    session = em.open_session(args.model, args.provider)
    inputs = em.model_inputs(session)
    if "scale" not in inputs:
        raise SystemExit("%s declares %s: there is no token to sweep" % (args.model, ", ".join(inputs)))
    out_name = session.get_outputs()[0].name

    keys = ("psnr_db", "ssim", "ms_ssim", "l1")
    rows = {token: {k: [] for k in keys} for token in tokens}
    detail = {token: [] for token in tokens}
    bilinear = {k: [] for k in keys}
    tiers = set()
    frames = 0

    for scene, entries in sorted(scenes.items()):
        em.log("scene %s (%d frames)" % (scene, len(entries)))
        for entry in entries:
            pair = em.load_pair(args.data, entry)
            low = pair["input"]
            target = em.to_image(pair["target"])
            # The frame's own sub-pixel phase, from the capture rather than defaulted: the released model
            # declares `jitter`, and zeros would be a claim about where the sample was taken.
            with np.load(os.path.join(args.data, entry["file"])) as raw:
                jitter = raw["jitter"] if "jitter" in raw.files else None
            feed = build_feed_from_image(low, inputs, jitter)
            tiers.add(round(float(np.log2(max(low.shape[1], 1) / 128.0)), 6))
            frames += 1

            up = em.to_image(pil_upscale(low, 2, Image.BILINEAR))
            for key, value in qm.evaluate(up, target).items():
                if key in bilinear:
                    bilinear[key].append(value)

            for token in tokens:
                feed["scale"] = np.full((1, 1, low.shape[0], low.shape[1]), np.float32(token), np.float32)
                candidate = em.to_image(session.run([out_name], feed)[0])
                for key, value in qm.evaluate(candidate, target).items():
                    if key in keys:
                        rows[token][key].append(value)
                detail[token].append(em.high_frequency_energy(candidate) /
                                     em.high_frequency_energy(target))

    report = {"data": args.data, "split": args.split, "model": args.model, "frames": frames,
              "tiers": sorted(tiers), "tokens": tokens,
              "bilinear": {k: em.accumulate(bilinear[k]) for k in keys},
              "by_token": {}}

    em.log("")
    em.log("%-10s %-9s %-8s %-9s %-8s %-8s" % ("token", "psnr_db", "ssim", "ms_ssim", "l1", "detail"))
    b = report["bilinear"]
    em.log("%-10s %-9.4f %-8.5f %-9.5f %-8.5f %-8s"
           % ("bilinear", b["psnr_db"]["mean"], b["ssim"]["mean"], b["ms_ssim"]["mean"], b["l1"]["mean"], "-"))
    for token in tokens:
        acc = {key: em.accumulate(rows[token][key]) for key in keys}
        det = em.accumulate(detail[token])
        own = "  <- this tier's own token" if any(abs(token - t) < 1e-6 for t in tiers) else ""
        em.log("%-10.4f %-9.4f %-8.5f %-9.5f %-8.5f %-8.4f%s"
               % (token, acc["psnr_db"]["mean"], acc["ssim"]["mean"], acc["ms_ssim"]["mean"],
                  acc["l1"]["mean"], det["mean"], own))
        report["by_token"]["%.6f" % token] = {
            "token": token, "own_tier": bool(own),
            "psnr_db": acc["psnr_db"]["mean"], "ssim": acc["ssim"]["mean"],
            "ms_ssim": acc["ms_ssim"]["mean"], "l1": acc["l1"]["mean"],
            "detail_ratio": det["mean"],
            "above_bilinear_db": acc["psnr_db"]["mean"] - b["psnr_db"]["mean"],
        }

    em.log("")
    em.log("%d frame(s), tier(s) present: %s"
           % (frames, ", ".join("log2 = %.4f" % t for t in sorted(tiers))))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        em.log("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
