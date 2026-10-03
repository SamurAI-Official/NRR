#!/usr/bin/env python3
"""Compare NRR against the upscalers a game actually ships, on the same held-out frames and the same metrics.

Methods: bilinear (the trainer's exact baseline), bicubic and Lanczos (the fixed-function no-neural floor),
AMD FSR 1.0 (the open-source spatial upscaler, ported in tools/fsr1.py), and the NRR model. Every method
receives the identical low-res input and is scored by the identical pipeline - image metrics (PSNR/SSIM/
MS-SSIM), perceptual metrics (LPIPS/DISTS), video quality (VMAF), detail retention, and temporal stability -
so the comparison cannot drift between code paths.

FSR 1.0 is the only commercial upscaler compared here. DLSS, XeSS and FSR 2/3/4 are temporal upscalers: they
integrate motion vectors, jitter and history across frames, so a fair comparison needs NRR's temporal path,
which is not trained yet. That gap is stated, not papered over.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as em  # noqa: E402  (reuses its loader, metrics and VMAF plumbing)
import fsr1                   # noqa: E402
import quality_metrics as qm  # noqa: E402

METHODS = ("bilinear", "bicubic", "lanczos", "fsr1", "nrr")


def pil_upscale(image, factor, resample):
    """Bicubic/Lanczos via Pillow's implementation - the fixed-function path a game uses without a neural
    upscaler. Operates on 8-bit display data, which is how these filters are actually shipped."""
    from PIL import Image
    u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = u8.shape[:2]
    out = np.asarray(Image.fromarray(u8, "RGB").resize((w * factor, h * factor), resample=resample))
    return out.astype(np.float32) / 255.0


def make_upscalers(model_onnx, provider):
    """Returns a dict name -> callable(image) -> upscaled image. `nrr` is present only when a model is given."""
    from PIL import Image

    def bicubic(image):
        return pil_upscale(image, 2, Image.BICUBIC)

    def lanczos(image):
        return pil_upscale(image, 2, Image.LANCZOS)

    def fsr1_u(image):
        return fsr1.fsr1_upscale(image, image.shape[0] * 2, image.shape[1] * 2)

    methods = {
        "bilinear": lambda image: em.bilinear_upscale(image, 2),
        "bicubic": bicubic,
        "lanczos": lanczos,
        "fsr1": fsr1_u,
    }
    if model_onnx:
        session = em.open_session(model_onnx, provider)
        inputs = em.model_inputs(session)
        out_name = session.get_outputs()[0].name

        def nrr(image):
            feed = build_feed_from_image(image, inputs)
            return em.to_image(session.run([out_name], feed)[0])

        methods["nrr"] = nrr
    return methods


def build_feed_from_image(image, inputs):
    """The ONNX feed for a bare colour image, with zeros for any optional inputs the model still declares."""
    feed = {}
    if "color" in inputs:
        feed["color"] = image.transpose(2, 0, 1)[None].astype(np.float32)
    if "depth" in inputs:
        feed["depth"] = np.zeros((1, 1, image.shape[0], image.shape[1]), np.float32)
    if "motion" in inputs:
        feed["motion"] = np.zeros((1, 2, image.shape[0], image.shape[1]), np.float32)
    if "history" in inputs:
        feed["history"] = np.zeros((1, 3, image.shape[0], image.shape[1]), np.float32)
    return feed


def main(argv):
    parser = argparse.ArgumentParser(description="Compare NRR against bilinear/bicubic/Lanczos/FSR 1.0.")
    parser.add_argument("--data", default="models/training-data/godot-v2")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0, help="cap frames per scene (0 = all)")
    parser.add_argument("--model", default="", help="the NRR .onnx to compare (omit to skip NRR)")
    parser.add_argument("--provider", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="device for LPIPS/DISTS backbones")
    parser.add_argument("--skip-perceptual", action="store_true")
    parser.add_argument("--skip-vmaf", action="store_true")
    parser.add_argument("--out", default="", help="where to write the JSON report")
    args = parser.parse_args(argv)

    manifest = em.load_manifest(args.data)
    scenes = em.group_sequences(manifest, args.split, args.limit)
    if not scenes:
        raise SystemExit("no %s scenes in %s" % (args.split, args.data))

    upscalers = make_upscalers(args.model or None, args.provider)
    methods = [m for m in METHODS if m in upscalers]

    perceptual_device = "cuda" if (args.device == "auto" and em.torch is not None and
                                   em.torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    perceptual = em.Perceptual(device=perceptual_device) if not args.skip_perceptual else None
    ffmpeg, has_libvmaf, _ = em.ffmpeg_probe()
    run_vmaf = not args.skip_vmaf and has_libvmaf

    report = {"data": args.data, "split": args.split, "methods": methods,
              "vmaf": has_libvmaf, "perceptual": bool(perceptual and perceptual.lpips is not None),
              "scenes": {}}

    image_keys = ("psnr_db", "ssim", "ms_ssim", "l1")
    with tempfile.TemporaryDirectory(prefix="nrr-compare-") as workdir:
        for scene, entries in sorted(scenes.items()):
            em.log("scene %s (%d frames): %s" % (scene, len(entries), ", ".join(methods)))
            frames = {m: [] for m in methods}
            metrics = {m: {k: [] for k in image_keys} for m in methods}
            detail = {m: [] for m in methods}
            perceptual_metrics = {m: {"lpips": [], "dists": []} for m in methods}

            for entry in entries:
                pair = em.load_pair(args.data, entry)
                inp = pair["input"]
                target = em.to_image(pair["target"])
                for m in methods:
                    up = upscalers[m](inp)
                    frames[m].append(up)
                    q = qm.evaluate(up, target)
                    for k in image_keys:
                        metrics[m][k].append(q[k])
                    detail[m].append(em.high_frequency_energy(up) / em.high_frequency_energy(target))
                    if perceptual is not None and (perceptual.lpips is not None or perceptual.dists is not None):
                        for name, value in perceptual.distances(up, target).items():
                            if value is not None:
                                perceptual_metrics[m][name].append(value)

            scene_report = {}
            target_seq = ([em.to_image(em.load_pair(args.data, e)["target"]) for e in entries]
                          if run_vmaf else None)
            for m in methods:
                scene_report[m] = {
                    "image": {k: em.accumulate(metrics[m][k]) for k in image_keys},
                    "detail_ratio": em.accumulate(detail[m]),
                }
                if not args.skip_perceptual:
                    scene_report[m]["perceptual"] = {
                        name: em.accumulate(perceptual_metrics[m][name])
                        for name in ("lpips", "dists")}
                if run_vmaf:
                    scene_report[m]["vmaf"] = em.vmaf_sequence(frames[m], target_seq, ffmpeg, workdir)
            report["scenes"][scene] = scene_report

    out_path = args.out or os.path.join(args.data, "compare_upscalers.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    _print_comparison(report)
    print("\nreport: %s" % out_path)
    return 0


def _mean(stats, ndigits=3):
    value = (stats or {}).get("mean")
    return "    -" if value is None else ("%*.3f" % (ndigits + 4, value))


def _print_comparison(report):
    methods = report["methods"]
    print("\n" + "=" * 100)
    for scene, scene_report in sorted(report["scenes"].items()):
        print("scene %s" % scene)
        print("%-10s %s %s %s %s %s %s %s" % ("method", "psnr", "ssim", "ms_ssim", "lpips", "dists", "detail", "vmaf"))
        for m in methods:
            r = scene_report[m]
            lp = (r.get("perceptual", {}) or {}).get("lpips")
            ds = (r.get("perceptual", {}) or {}).get("dists")
            vmaf = (r.get("vmaf", {}) or {}).get("vmaf_mean")
            print("%-10s %s %s %s %s %s %s %s" % (
                m, _mean(r["image"]["psnr_db"]), _mean(r["image"]["ssim"]),
                _mean(r["image"]["ms_ssim"]), _mean(lp), _mean(ds),
                _mean(r["detail_ratio"]), _mean({"mean": vmaf})))
        print("-" * 100)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

