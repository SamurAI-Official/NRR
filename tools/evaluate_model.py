#!/usr/bin/env python3
"""Evaluate an exported NRR upscaler across the dimensions a commercial upscaler is compared on.

| dimension               | metric                                | source                          |
| ----------------------- | ------------------------------------- | ------------------------------- |
| reconstruction fidelity | PSNR                                  | tools/quality_metrics.py (C++-pinned) |
| structural              | SSIM, MS-SSIM                         | tools/quality_metrics.py        |
| perceptual              | LPIPS, DISTS                          | lpips, DISTS_pytorch            |
| video quality           | VMAF, plus its motion feature         | ffmpeg libvmaf / vmafmotion     |
| temporal stability      | warping error, temporal PSNR/SSIM     | tools/quality_metrics.py        |
| detail retention        | high-frequency energy ratio           | computed here                   |
| performance             | per-tier latency, ms/frame            | tools/measure_model.py          |
| robustness              | per-scene spread, not a single mean   | computed here                   |

Two rules the report follows, because both have bitten this project before:

* Every image metric is reported for the model **and** for the bilinear baseline on the same frames. "Better"
  then means better than the thing it replaces, not better than nothing.
* Temporal metrics are reported beside the reference's own. A reconstruction that is smoother than the
  ground truth is not more stable, only blurrier, and a single number cannot tell those apart.

Nothing is silently skipped. A metric whose dependency is missing appears with `available: false` and the
reason, so a short report cannot be mistaken for a clean one. The motion convention is the capture's
(`tools/godot_capture/shaders/motion.gdshader`: `motion = cur_uv - prev_uv`, viewport UV, +y down), read
from the manifest's own description rather than assumed, and the frames are used in capture order because
temporal metrics on shuffled frames measure nothing.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# torch is imported lazily as an optional dependency: the CI self-test and the metric core in
# quality_metrics.py need only numpy, and importing torch (a multi-GB wheel) on a runner that will never
# use it would make the harness ungateable. Anything that needs torch (bilinear_upscale, the perceptual
# backbones, the --device auto choice) refuses with a clear message when it is absent.
try:
    import torch
except ImportError:                                   # pragma: no cover - exercised on the CI runner
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import quality_metrics as qm  # noqa: E402  (path set above; the PSNR/SSIM mirror of the runtime)

# The channels each named model input carries, matching train_nrr.py's INPUT_NAMES.
INPUT_CHANNELS = {"color": 3, "depth": 1, "motion": 2, "history": 3}


def log(message):
    print(message, flush=True)


def load_manifest(data_dir):
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit("%s has no manifest.json - run tools/gen_training_pairs.py first" % data_dir)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def group_sequences(manifest, split, limit):
    """Frames grouped by scene and sorted by capture order.

    A sequence is what temporal metrics and VMAF need. The captures are consecutive within a scene, so the
    order is the capture order rather than a choice, and sorting by the recorded frame index is what makes
    the warp between consecutive frames valid."""
    scenes = {}
    for entry in manifest["pairs"]:
        if entry["split"] != split:
            continue
        scenes.setdefault(entry["scene"], []).append(entry)
    for name in list(scenes):
        scenes[name].sort(key=lambda entry: entry["frame"])
        if limit:
            scenes[name] = scenes[name][:limit]
        if len(scenes[name]) < 2:
            # A one-frame scene cannot contribute a temporal measurement, and pretending it can by
            # duplicating the frame would report perfect stability for a video nobody rendered.
            del scenes[name]
    return scenes


def open_session(model_path, provider):
    import onnxruntime
    available = onnxruntime.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise SystemExit("CUDAExecutionProvider is unavailable; onnxruntime offers %s" % available)
    if provider == "cpu" or "CUDAExecutionProvider" not in available:
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return onnxruntime.InferenceSession(model_path, providers=providers)


def model_inputs(session):
    return [node.name for node in session.get_inputs()]


def load_pair(data_dir, entry):
    """One captured pair as numpy, plus the raw arrays the metrics need.

    The noisy input is the default because that is what the model sees in the trainer and in the runtime's
    real frame path. `input_clean` exists in the capture and is offered behind a flag for the same reason the
    trainer offers it, but it is not the product input."""
    with np.load(os.path.join(data_dir, entry["file"])) as pair:
        return {
            "input": pair["input"].astype(np.float32),
            "input_clean": pair["input_clean"].astype(np.float32) if "input_clean" in pair else None,
            "target": pair["target"].astype(np.float32),
            "depth": pair["depth"].astype(np.float32),
            "motion": pair["motion"].astype(np.float32),
            "history": pair["history"].astype(np.float32) if "history" in pair else None,
            # The per-frame sub-pixel sampling offset, kept as the raw 2-vector. `build_feed` is what
            # broadcasts it, and it must match the trainer's layout exactly - a model that scored differently
            # here than it trained would make every downstream comparison a lie.
            "jitter": pair["jitter"].astype(np.float32) if "jitter" in pair else None,
        }


def build_feed(pair, inputs, use_clean):
    """The ONNX feed for one frame, using exactly the names and layout the trainer exported."""
    color = pair["input_clean"] if (use_clean and pair["input_clean"] is not None) else pair["input"]
    feed = {}
    if "color" in inputs:
        feed["color"] = color.transpose(2, 0, 1)[None].copy()
    if "depth" in inputs:
        feed["depth"] = pair["depth"][None, None].copy()
    if "motion" in inputs:
        feed["motion"] = pair["motion"].transpose(2, 0, 1)[None].copy()
    if "history" in inputs:
        history = pair["history"] if pair["history"] is not None else np.zeros_like(color)
        feed["history"] = history.transpose(2, 0, 1)[None].copy()
    if "jitter" in inputs:
        # The 2-vector is broadcast to a plane for the same reason the trainer broadcasts it: the graph's
        # de-jitter stage builds a sampling grid the size of the frame, so it needs a full H x W offset plane,
        # not a single vector. `np.broadcast_to` is a view, hence the copy - the session needs a real buffer.
        offset = pair["jitter"] if pair.get("jitter") is not None else np.zeros(2, np.float32)
        feed["jitter"] = np.ascontiguousarray(
            np.broadcast_to(offset.reshape(1, 2, 1, 1), (1, 2, color.shape[0], color.shape[1])))
    return feed


def to_image(array):
    """(1,3,H,W) or (3,H,W) or (H,W,3) float -> (H,W,3) float32 in [0,1]."""
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def bilinear_upscale(image, factor=2):
    """The baseline the model must beat, computed the way the model's own skip is
    (train_nrr.baseline_upscale: bilinear, align_corners=False)."""
    if torch is None:
        raise RuntimeError("bilinear_upscale needs torch, which is not installed on this machine")
    tensor = torch.from_numpy(image.transpose(2, 0, 1)[None].copy())
    up = torch.nn.functional.interpolate(tensor, scale_factor=factor, mode="bilinear", align_corners=False)
    return up[0].numpy().transpose(1, 2, 0)


LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def high_frequency_energy(image):
    """Mean absolute Laplacian - a cheap stand-in for the detail a viewer reads.

    The ratio model/reference is what matters, not the absolute value: 1.0 is as much detail as the truth,
    below 1.0 is softening, and above 1.0 is detail the input never had, which on an upscaler usually means
    ringing or over-sharpening rather than recovered structure.

    This one number cannot tell recovered detail from aliasing, and on an aliased input it cannot: a real
    low-resolution raster carries energy above its own Nyquist, an upscaler smears that energy into the
    output, and a Laplacian counts it exactly as it counts recovered structure. `detail_bands` splits the two
    apart. The Laplacian is kept because it is what every earlier number in this project was measured with -
    changing it silently would make those numbers incomparable - not because it is the better measurement."""
    grey = image @ LUMA
    laplacian = np.zeros_like(grey)
    laplacian[1:-1, 1:-1] = (4.0 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1]
                             - grey[1:-1, :-2] - grey[1:-1, 2:])
    return float(np.mean(np.abs(laplacian)))


def band_component(image, lo, hi):
    """The component of an image between `lo` and `hi` cycles per *output* pixel.

    Radially band-limited in the frequency domain, so "the detail in this band" means the detail at those
    spatial frequencies and nothing else. Returned with the mean removed and in the input's own units, so
    two images of the same size can be compared directly."""
    grey = (image @ LUMA).astype(np.float64)
    grey = grey - grey.mean()
    height, width = grey.shape
    spectrum = np.fft.fftshift(np.fft.fft2(grey))
    y, x = np.indices((height, width))
    # Bin offsets are frequency * height, so dividing by height makes `radius` cycles per pixel directly and
    # puts the axis Nyquist at 0.5 - which is what the band edges are written against. Dividing by height/2
    # instead puts Nyquist at 1.0, which silently makes the "unrecoverable" band measure frequencies the
    # input could in fact carry; the self-test's sinusoid checks are what caught that.
    radius = np.sqrt((y - height // 2) ** 2 + (x - width // 2) ** 2) / float(height)
    spectrum[(radius < lo) | (radius >= hi)] = 0.0
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def recoverable_edge(factor):
    """The highest frequency, in cycles per output pixel, that the low-resolution input can carry.

    A `factor`x upscale puts the input's Nyquist at 0.5/factor cycles per *output* pixel: content above it
    was never in the input and no single-frame upscaler can have recovered it. Derived from the actual
    factor rather than hardcoded, so a 3x dataset moves the edge instead of being measured with a 2x
    assumption."""
    return 0.5 / max(int(factor), 1)


def detail_bands(image, edge):
    """(recoverable, unrecoverable) mean **absolute** band amplitude, split at `edge`.

    Mean absolute rather than mean square, deliberately: `high_frequency_energy` above is a mean-absolute
    Laplacian, so this is the same statistic split by frequency, and the two are directly comparable. A
    mean-square version was tried first and was misleading here - one model has a handful of ringing pixels,
    and mean square reports that handful as a 5-9x failure in the band above Nyquist while the typical pixel
    is unchanged. Ringing is real and worth reporting, so it gets its own number (`ringing_index`) instead of
    distorting this one.

    The split separates two questions the Laplacian conflates: whether the reconstruction carries the detail
    the input could carry, and whether it carries anything the input could not."""
    recoverable = band_component(image, 0.0, edge)
    unrecoverable = band_component(image, edge, 0.5)
    return float(np.mean(np.abs(recoverable))), float(np.mean(np.abs(unrecoverable)))


def ringing_index(image, edge):
    """Mean-square energy above the input's Nyquist, as a count of spikes rather than a bulk statistic.

    A mean-absolute statistic barely notices a few over-driven pixels; this notices them by construction, and
    a value well above 1.0 means the reconstruction is putting energy where the input had none. Reported,
    never gated: a little is normal edge contrast and a lot is ringing, and where that line sits is a
    judgement this harness is not qualified to make silently."""
    return float(np.mean(band_component(image, edge, 0.5) ** 2))


def band_correlation(image, reference, edge):
    """Correlation of the recoverable band of `image` with the same band of `reference`.

    Energy alone cannot distinguish the *right* detail from the wrong detail at the same amplitude, which is
    the difference between recovering structure and amplifying aliasing into a plausible-looking pattern.
    This asks whether the detail in the band the input could carry actually looks like the truth's."""
    a = band_component(image, 0.0, edge).ravel()
    b = band_component(reference, 0.0, edge).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        # A band with no variance carries no structure to agree or disagree about; reporting 0.0 would read
        # as "uncorrelated" when it means "nothing to correlate".
        return None
    return float(np.corrcoef(a, b)[0, 1])


class Perceptual:
    """LPIPS and DISTS, loaded once per run.

    Each reports its own unavailability rather than being dropped from the report: a metric that failed to
    load and a metric that was never asked for look identical in a table of numbers, and only one of them is
    a reason to distrust the comparison. The backbones are pretrained on natural images, so the numbers are
    comparable with published ones only in so far as the content is - stated here so a low LPIPS is not read
    as "better than DLSS" on a scene neither was trained on."""

    def __init__(self, device="cpu"):
        self.device = device
        self.lpips = None
        self.dists = None
        self.reasons = {}
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:                      # noqa: BLE001 - any failure is reported, not raised
            self.reasons["lpips"] = "%s: %s" % (type(exc).__name__, exc)
        try:
            from DISTS_pytorch import DISTS
            self.dists = DISTS().to(device).eval()
        except Exception as exc:                      # noqa: BLE001
            self.reasons["dists"] = "%s: %s" % (type(exc).__name__, exc)

    def _pair(self, a, b):
        ta = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)[None])).to(self.device)
        tb = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)[None])).to(self.device)
        return ta, tb

    def distances(self, a, b):
        """LPIPS and DISTS for one pair, either of which may be None if its package is missing.

        Both are run under no_grad on the same moved-to-device tensors, and neither mutates its input."""
        out = {"lpips": None, "dists": None}
        if self.lpips is None and self.dists is None:
            return out
        ta, tb = self._pair(a, b)
        with torch.no_grad():
            if self.lpips is not None:
                out["lpips"] = float(self.lpips(ta, tb))
            if self.dists is not None:
                out["dists"] = float(self.dists(ta, tb))
        return out


def ffmpeg_probe():
    """(path, has_libvmaf, has_vmafmotion). All False when ffmpeg is absent, which is reported, not faked."""
    path = shutil.which("ffmpeg")
    if path is None:
        return None, False, False
    try:
        listing = subprocess.run([path, "-hide_banner", "-filters"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace").stdout
    except OSError:
        return path, False, False
    return path, ("libvmaf" in listing), ("vmafmotion" in listing)


def _write_raw(images, path):
    with open(path, "wb") as handle:
        for image in images:
            handle.write(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())


def vmaf_sequence(distorted, reference, ffmpeg, workdir):
    """Pooled VMAF of a distorted sequence against a reference sequence, via ffmpeg's libvmaf.

    The frames travel as raw RGB24 files rather than as PNGs, so the metric does not depend on an image
    codec's own choices, and ffmpeg is run with its working directory set to the scratch directory so the
    filter's `log_path` never has to carry a Windows drive colon - escaping that through a filtergraph is a
    known source of silently wrong runs. A failure raises with ffmpeg's own message, because a VMAF of zero
    and a VMAF that never ran must not look the same."""
    if len(distorted) != len(reference) or not distorted:
        raise RuntimeError("VMAF needs two equal-length non-empty sequences")
    height, width = reference[0].shape[:2]
    size = "%dx%d" % (width, height)
    _write_raw(distorted, os.path.join(workdir, "dist.rgb"))
    _write_raw(reference, os.path.join(workdir, "ref.rgb"))
    log_name = "vmaf.json"
    log_path = os.path.join(workdir, log_name)
    if os.path.exists(log_path):
        os.remove(log_path)
    graph = ("[0:v]format=yuv420p[dist];[1:v]format=yuv420p[ref];"
             "[dist][ref]libvmaf=log_fmt=json:log_path=%s" % log_name)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", size, "-framerate", "60",
               "-i", "dist.rgb",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", size, "-framerate", "60",
               "-i", "ref.rgb",
               "-lavfi", graph, "-f", "null", "-"]
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0 or not os.path.exists(log_path):
        raise RuntimeError("libvmaf failed (%d): %s" % (result.returncode, (result.stderr or "").strip()[-400:]))
    with open(log_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    pooled = payload.get("pooled_metrics", {})
    if "vmaf" not in pooled:
        raise RuntimeError("libvmaf returned no vmaf metric: %s" % sorted(pooled))
    score = pooled["vmaf"]
    return {"vmaf_mean": score.get("mean"), "vmaf_min": score.get("min"), "vmaf_max": score.get("max"),
            "frames": len(payload.get("frames", distorted))}


def vmafmotion_score(reference, ffmpeg, workdir):
    """The VMAF motion feature of a reference sequence, as a scalar. Returns None when ffmpeg does not print
    a parsable value, rather than defaulting to zero."""
    try:
        height, width = reference[0].shape[:2]
        _write_raw(reference, os.path.join(workdir, "motion_ref.rgb"))
        command = [ffmpeg, "-hide_banner", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", "%dx%d" % (width, height), "-framerate", "60", "-i", "motion_ref.rgb",
                   "-lavfi", "vmafmotion", "-f", "null", "-"]
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        for line in (result.stderr or "").splitlines():
            lowered = line.lower()
            if "motion avg" in lowered:
                return float(line.split(":")[-1].strip())
        return None
    except Exception:                                 # noqa: BLE001 - a missing extra is not a failure
        return None


def accumulate(values):
    """Mean/σ/min/max of the non-None values, with the count. σ is reported so a claim is a spread, not a
    single number - a mean over 200 frames with σ the same size as the effect is noise, and the report says
    so rather than hiding it."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(present, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def evaluate_scene(session, inputs, entries, data_dir, perceptual, ffmpeg, workdir, use_clean,
                   run_perceptual, run_vmaf):
    """Every metric for one scene, returned as a dict. Image metrics for model and baseline, temporal metrics
    for model and reference, detail-retention ratios, and VMAF when the pieces are present."""
    output_name = session.get_outputs()[0].name
    frames = []
    for entry in entries:
        pair = load_pair(data_dir, entry)
        feed = build_feed(pair, inputs, use_clean)
        output = to_image(session.run([output_name], feed)[0])
        target = to_image(pair["target"])
        source = pair["input_clean"] if (use_clean and pair["input_clean"] is not None) else pair["input"]
        baseline = bilinear_upscale(source)
        factor = target.shape[0] // max(source.shape[0], 1)
        frames.append({"output": output, "baseline": baseline, "target": target,
                       "factor": factor,
                       "motion_up": qm.upscale_motion_nearest(pair["motion"], factor)})

    image = {key: {"model": [], "baseline": []} for key in ("psnr_db", "ssim", "ms_ssim", "l1")}
    perceptual_metrics = {"lpips": {"model": [], "baseline": []}, "dists": {"model": [], "baseline": []}}
    detail = {"model": [], "baseline": []}
    # The band split. `factor` is the upscale actually used for this dataset, so the edge follows the data
    # rather than assuming 2x. Everything here is reported *beside* the Laplacian ratio, never instead of it.
    edge = recoverable_edge(frames[0]["factor"])
    bands = {"recoverable": {"model": [], "baseline": []},
             "unrecoverable": {"model": [], "baseline": []},
             "correlation": {"model": [], "baseline": []}}
    ringing = {"model": [], "baseline": []}
    temporal = {key: {"model": [], "reference": []}
                for key in ("warping_error", "temporal_psnr_db", "temporal_ssim")}
    verified = []
    for frame in frames:
        target_bands = detail_bands(frame["target"], edge)
        target_ring = max(ringing_index(frame["target"], edge), 1e-20)
        for key, target_key in (("model", "output"), ("baseline", "baseline")):
            image_bands = detail_bands(frame[target_key], edge)
            bands["recoverable"][key].append(image_bands[0] / max(target_bands[0], 1e-20))
            bands["unrecoverable"][key].append(image_bands[1] / max(target_bands[1], 1e-20))
            bands["correlation"][key].append(band_correlation(frame[target_key], frame["target"], edge))
            ringing[key].append(ringing_index(frame[target_key], edge) / target_ring)
            metrics = qm.evaluate(frame[target_key], frame["target"])
            for name in image:
                image[name][key].append(metrics[name])
            detail[key].append(high_frequency_energy(frame[target_key]) / high_frequency_energy(frame["target"]))
            if run_perceptual:
                for name, value in perceptual.distances(frame[target_key], frame["target"]).items():
                    if value is not None:
                        perceptual_metrics[name][key].append(value)

    for previous, current in zip(frames, frames[1:]):
        for key, prev_name, cur_name in (("model", "output", "output"),
                                         ("reference", "target", "target")):
            result = qm.temporal_stability(previous[prev_name], current[cur_name], current["motion_up"])
            for name in temporal:
                temporal[name][key].append(result[name])
            verified.append(result["verified_fraction"])

    report = {
        "frames": len(frames),
        "image": {name: {key: accumulate(image[name][key]) for key in image[name]} for name in image},
        "detail_ratio": {key: accumulate(detail[key]) for key in detail},
        # The band split, reported beside the Laplacian ratio above and never instead of it. `edge` is
        # recorded so a reader can see which frequencies count as recoverable for this dataset's upscale.
        "detail_bands": {"edge_cycles_per_output_pixel": edge,
                         **{name: {key: accumulate(bands[name][key]) for key in bands[name]}
                            for name in bands},
                         "ringing": {key: accumulate(ringing[key]) for key in ringing}},
        "temporal": {name: {key: accumulate(temporal[name][key]) for key in temporal[name]}
                     for name in temporal},
        "temporal_verified_fraction": accumulate(verified),
    }
    if run_perceptual:
        report["perceptual"] = {name: {key: accumulate(perceptual_metrics[name][key])
                                       for key in perceptual_metrics[name]}
                                for name in perceptual_metrics}
    if run_vmaf and ffmpeg is not None:
        report["vmaf"] = {
            "model": vmaf_sequence([f["output"] for f in frames], [f["target"] for f in frames],
                                   ffmpeg, workdir),
            "baseline": vmaf_sequence([f["baseline"] for f in frames], [f["target"] for f in frames],
                                      ffmpeg, workdir),
        }
        motion = vmafmotion_score([f["target"] for f in frames], ffmpeg, workdir)
        if motion is not None:
            report["vmaf"]["reference_motion"] = motion
    return report


def _stat(stats, key="mean", ndigits=4):
    value = (stats or {}).get(key)
    if value is None:
        return "     -"
    if key == "mean":
        return "%*.4f" % (ndigits + 5, value)
    return "%*g" % (ndigits + 5, value)


def print_summary(report):
    log("")
    for scene, scene_report in sorted(report["scenes"].items()):
        image = scene_report["image"]
        log("scene %-10s frames=%d  psnr %s dB (base %s) | ssim %s (base %s) | ms_ssim %s (base %s)"
            % (scene, scene_report["frames"],
               _stat(image["psnr_db"]["model"]), _stat(image["psnr_db"]["baseline"]),
               _stat(image["ssim"]["model"]), _stat(image["ssim"]["baseline"]),
               _stat(image["ms_ssim"]["model"]), _stat(image["ms_ssim"]["baseline"])))
        detail = scene_report.get("detail_ratio", {})
        log("  detail ratio   model %s | baseline %s   (1.0 = as much detail as the truth)"
            % (_stat(detail.get("model")), _stat(detail.get("baseline"))))
        split = scene_report.get("detail_bands", {})
        if split:
            edge = split.get("edge_cycles_per_output_pixel")
            log("  detail bands   edge %.3f cyc/px (the input's Nyquist; above it the input carried nothing)"
                % edge)
            log("    recoverable  model %s | baseline %s   (energy vs the truth's, input could carry this)"
                % (_stat(split["recoverable"].get("model")), _stat(split["recoverable"].get("baseline"))))
            log("    uncarr.      model %s | baseline %s   (energy vs truth's; >1 is aliasing, not detail)"
                % (_stat(split["unrecoverable"].get("model")), _stat(split["unrecoverable"].get("baseline"))))
            log("    correlation  model %s | baseline %s   (recoverable band vs the truth's own)"
                % (_stat(split["correlation"].get("model")), _stat(split["correlation"].get("baseline"))))
            if "ringing" in split:
                log("    ringing      model %s | baseline %s   (mean-square above Nyquist vs truth's)"
                    % (_stat(split["ringing"].get("model")), _stat(split["ringing"].get("baseline"))))
        temporal = scene_report.get("temporal", {})
        verified = (scene_report.get("temporal_verified_fraction", {}).get("mean") or 0.0) * 100.0
        log("  warping error  model %s | reference %s   (motion verified over %.0f%% of pixels)"
            % (_stat(temporal["warping_error"]["model"]), _stat(temporal["warping_error"]["reference"]),
               verified))
        log("  temporal ssim  model %s | reference %s"
            % (_stat(temporal["temporal_ssim"]["model"]), _stat(temporal["temporal_ssim"]["reference"])))
        if "perceptual" in scene_report:
            per = scene_report["perceptual"]
            log("  perceptual     lpips %s (base %s) | dists %s (base %s)"
                % (_stat(per["lpips"]["model"]), _stat(per["lpips"]["baseline"]),
                   _stat(per["dists"]["model"]), _stat(per["dists"]["baseline"])))
        if "vmaf" in scene_report:
            vmaf = scene_report["vmaf"]
            log("  vmaf           model %s | baseline %s   (reference motion %s)"
                % (_stat(vmaf["model"], "vmaf_mean"), _stat(vmaf["baseline"], "vmaf_mean"),
                   vmaf.get("reference_motion")))
            scenes[name] = scenes[name][:limit]
        if len(scenes[name]) < 2:
            # A one-frame scene cannot contribute a temporal measurement, and pretending it can by
            # duplicating the frame would report perfect stability for a video nobody rendered.
            del scenes[name]
    return scenes
def self_test():
    """The numpy-only checks that gate the harness on CI, where torch, ffmpeg, the model and the dataset are
    all absent. It pins the measurement *logic* - accumulation, the detail proxy, sequence ordering, tensor
    normalisation - which is the part of a harness whose wrongness would silently corrupt every number above
    it. The heavy metrics (perceptual, VMAF) are exercised on a machine that has them, not faked here."""
    failures = 0
    total = 0

    def check(name, condition):
        nonlocal failures, total
        total += 1
        print("%s: %s" % (name, "OK" if condition else "FAIL"))
        failures += 0 if condition else 1

    stats = accumulate([1.0, 2.0, 3.0, 4.0])
    check("accumulate mean is exact", abs(stats["mean"] - 2.5) < 1e-12)
    check("accumulate counts every value", stats["n"] == 4)
    check("accumulate reports sigma, not just a mean", stats["std"] == 1.118033988749895)
    empty = accumulate([])
    check("accumulate of nothing reports n=0, not a fake zero", empty["n"] == 0 and empty["mean"] is None)
    none_stats = accumulate([None, 1.0])
    check("accumulate skips None rather than averaging it", none_stats["n"] == 1 and none_stats["mean"] == 1.0)

    flat = np.full((16, 16, 3), 0.5, np.float32)
    check("detail of a flat image is zero", high_frequency_energy(flat) == 0.0)
    check("detail is strictly positive for a checkerboard",
          high_frequency_energy((np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32).repeat(3).reshape(16, 16, 3)) > 0.0)

    # The band split. These pin the part of the measurement that decides whether an aliased baseline is
    # being credited with detail it cannot have had, so they check which frequencies land in which band
    # rather than only that the functions return a number.
    check("a 2x upscale puts the input's Nyquist at 0.25 cycles/output-pixel", recoverable_edge(2) == 0.25)
    check("a 3x upscale moves the edge, it is not hardcoded", recoverable_edge(3) == 0.5 / 3)
    check("a 1x upscale does not divide by zero", recoverable_edge(1) == 0.5)
    check("a flat image has no energy in any band",
          detail_bands(flat, 0.25) == (0.0, 0.0))

    size = 128
    axis = np.arange(size, dtype=np.float64)

    def sinusoid(cycles, phase=0.0):
        """A greyscale image holding one pure spatial frequency.

        `cycles` is a whole number of cycles across the frame, so the signal is periodic over the window and
        does not leak into neighbouring bins. A frequency that is not a whole number of cycles (0.4 at this
        size is 51.2 cycles) discontinuously wraps at the frame edge and smears ~0.1% of its energy into
        every other band, which would make a perfectly correct filter look leaky.
        """
        wave = np.sin(2.0 * np.pi * cycles * axis / size + phase)[None, :].repeat(size, 0)
        return np.repeat(wave[:, :, None], 3, axis=2)

    low = band_component(sinusoid(16), 0.0, 0.25)          # 0.125 cyc/px, inside
    high = band_component(sinusoid(40), 0.0, 0.25)         # 0.3125 cyc/px, outside
    check("an in-band sinusoid survives the recoverable band",
          float(np.std(low)) > 0.9 * float(np.std(sinusoid(16))))
    check("an in-band sinusoid is removed by a disjoint band",
          float(np.std(band_component(sinusoid(16), 0.30, 0.5))) < 1e-12)
    check("an above-Nyquist sinusoid is refused by the recoverable band", float(np.std(high)) < 1e-12)
    check("...and is what the unrecoverable band is for",
          float(np.std(band_component(sinusoid(40), 0.25, 0.5))) > 0.9 * float(np.std(sinusoid(40))))
    # Aliasing and detail look identical to a Laplacian; the band split is what separates them. This is the
    # case the whole measurement exists for, so it is pinned directly rather than only through the helpers.
    aliasing = detail_bands(sinusoid(40), 0.25)
    detail_only = detail_bands(sinusoid(16), 0.25)
    check("a Laplacian counts above-Nyquist energy as detail just the same",
          high_frequency_energy(sinusoid(40)) > high_frequency_energy(sinusoid(16)) * 0.5)
    check("the band split routes it to the unrecoverable band instead",
          aliasing[1] > detail_only[1] * 10.0 and aliasing[0] < detail_only[0])
    check("correlation is 1.0 for a band matched with itself",
          abs(band_correlation(sinusoid(16), sinusoid(16), 0.25) - 1.0) < 1e-9)
    check("correlation is low for unrelated detail in the band",
          band_correlation(sinusoid(16, 0.0), sinusoid(17, 0.0), 0.25) < 0.5)
    check("correlation of a flat band reports None, not a fake 0.0",
          band_correlation(flat, flat, 0.25) is None)
    check("a band with no variance does not fabricate an anticorrelation",
          band_correlation(flat, sinusoid(16), 0.25) is None)
    # Ringing is spikes, so the statistic that finds it is mean-square over the band above Nyquist, and it
    # must stay quiet for a band-limited image carrying the same typical amplitude.
    clean = sinusoid(16)
    spikes = clean.copy()
    spikes[64, 64] = 40.0                      # one over-driven pixel, as an edge artefact looks
    check("a single spike barely moves the mean-absolute band statistic",
          detail_bands(spikes, 0.25)[0] < detail_bands(clean, 0.25)[0] * 1.05)
    check("...and is unmistakable in the ringing index",
          ringing_index(spikes, 0.25) > 100.0 * max(ringing_index(clean, 0.25), 1e-12))
    check("a flat image has no ringing", ringing_index(flat, 0.25) == 0.0)

    manifest = {"pairs": [
        {"file": "a.npz", "split": "val", "scene": "s1", "frame": 2},
        {"file": "b.npz", "split": "val", "scene": "s1", "frame": 0},
        {"file": "c.npz", "split": "val", "scene": "s1", "frame": 1},
        {"file": "d.npz", "split": "val", "scene": "solo", "frame": 0},
        {"file": "e.npz", "split": "train", "scene": "s1", "frame": 0},
    ]}
    scenes = group_sequences(manifest, "val", 0)
    check("only the requested split is kept", set(scenes) == {"s1"})
    check("frames are in capture order", [e["frame"] for e in scenes["s1"]] == [0, 1, 2])
    check("a one-frame scene is dropped, not faked", "solo" not in scenes)
    check("--limit caps frames per scene", len(group_sequences(manifest, "val", 2)["s1"]) == 2)

    rng = np.random.RandomState(0)
    chw = rng.rand(3, 4, 5).astype(np.float32)
    check("to_image(NCHW) lands on (H,W,3)", to_image(chw[None]).shape == (4, 5, 3))
    check("to_image(HWC) is unchanged", to_image(chw.transpose(1, 2, 0)).shape == (4, 5, 3))
    check("to_image clips into [0,1]", float(to_image(np.array([2.0], np.float32))[0]) == 1.0)
    check("to_image of (3,H,W) transposes to HWC", to_image(chw).shape == (4, 5, 3))

    feed = build_feed({"input": rng.rand(8, 8, 3).astype(np.float32),
                       "depth": rng.rand(8, 8).astype(np.float32),
                       "motion": rng.rand(8, 8, 2).astype(np.float32),
                       "history": rng.rand(8, 8, 3).astype(np.float32),
                       "input_clean": rng.rand(8, 8, 3).astype(np.float32)},
                      ["color", "depth", "motion", "history"], use_clean=False)
    check("build_feed lays color out NCHW", feed["color"].shape == (1, 3, 8, 8))
    check("build_feed keeps motion as two channels", feed["motion"].shape == (1, 2, 8, 8))
    check("build_feed respects --use-clean",
          build_feed({"input": np.zeros((8, 8, 3), np.float32),
                      "input_clean": np.ones((8, 8, 3), np.float32)},
                     ["color"], use_clean=True)["color"].min() == 1.0)

    print("RESULT: %s (%d checks)" % ("PASS" if failures == 0 else "FAIL", total))
    return 1 if failures else 0


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate an exported NRR upscaler across quality dimensions.")
    parser.add_argument("--model", default=None, help="the exported .onnx to evaluate (not needed for --self-test)")
    parser.add_argument("--data", default="models/training-data/godot-v2")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0, help="cap frames per scene (0 = all)")
    parser.add_argument("--use-clean", action="store_true", help="feed input_clean instead of the noisy input")
    parser.add_argument("--provider", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="device for LPIPS/DISTS backbones (auto = cuda when torch reports it)")
    parser.add_argument("--skip-perceptual", action="store_true")
    parser.add_argument("--skip-vmaf", action="store_true")
    parser.add_argument("--latency", action="store_true",
                        help="also run tools/measure_model.py and embed its per-tier latency")
    parser.add_argument("--out", default="", help="where to write the JSON report (default: beside the model)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the numpy-only metric-logic checks and exit (no model, dataset, torch, "
                             "ffmpeg or GPU needed - this is what CI runs)")
    return parser


def main(argv):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.model:
        raise SystemExit("--model is required (or run --self-test)")
    if not os.path.exists(args.model):
        raise SystemExit("model not found: %s" % args.model)

    manifest = load_manifest(args.data)
    scenes = group_sequences(manifest, args.split, args.limit)
    if not scenes:
        raise SystemExit("no %s scenes with at least two frames in %s" % (args.split, args.data))

    session = open_session(args.model, args.provider)
    inputs = model_inputs(session)

    ffmpeg, has_libvmaf, has_vmafmotion = ffmpeg_probe()
    run_vmaf = not args.skip_vmaf and has_libvmaf
    run_perceptual = not args.skip_perceptual
    perceptual_device = args.device
    if perceptual_device == "auto":
        perceptual_device = "cuda" if torch.cuda.is_available() else "cpu"
    # The backbones download on first use and cost a few seconds to load, so they are only loaded when the
    # run actually asks for them; --skip-perceptual must not pay for a comparison it refuses to make.
    perceptual = Perceptual(device=perceptual_device) if run_perceptual else None

    report = {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "model_inputs": inputs,
        "perceptual_device": perceptual_device if run_perceptual else None,
        "capabilities": {"lpips": bool(perceptual and perceptual.lpips is not None),
                         "dists": bool(perceptual and perceptual.dists is not None),
                         "vmaf": has_libvmaf,
                         "vmafmotion": has_vmafmotion},
        "unavailable": dict(perceptual.reasons) if perceptual else {},
        "scenes": {},
    }
    if not has_libvmaf and not args.skip_vmaf:
        report["unavailable"]["vmaf"] = "ffmpeg is missing or has no libvmaf filter"
    if not run_perceptual:
        report["unavailable"]["perceptual"] = "disabled by --skip-perceptual"

    with tempfile.TemporaryDirectory(prefix="nrr-eval-") as workdir:
        for scene in sorted(scenes):
            log("evaluating scene %s (%d frames)" % (scene, len(scenes[scene])))
            report["scenes"][scene] = evaluate_scene(
                session, inputs, scenes[scene], args.data, perceptual,
                ffmpeg if run_vmaf else None, workdir, args.use_clean, run_perceptual, run_vmaf)

    if args.latency:
        measure = os.path.join(HERE, "measure_model.py")
        result = subprocess.run([sys.executable, measure, "--model", args.model, "--out", ""],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            report["latency"] = {"tool": "tools/measure_model.py", "note": "see its report beside the model"}
        else:
            report["unavailable"]["latency"] = (result.stderr or "").strip()[-300:]

    out_path = args.out or (os.path.splitext(args.model)[0] + ".eval.json")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print_summary(report)
    log("\nreport: %s" % out_path)
    if report["unavailable"]:
        log("unavailable: %s" % json.dumps(report["unavailable"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))