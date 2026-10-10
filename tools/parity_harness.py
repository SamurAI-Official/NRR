#!/usr/bin/env python3
"""parity_harness.py - the head-to-head table, produced by one command.

`docs/parity.md` is the instrument the "DLSS/XeSS parity" claim rests on, and the reason it needs its
own harness rather than another run of `compare_upscalers.py` is a property of the competitors:

    DLSS and XeSS cannot be run offline from a `.npz`. Each needs its own device, its own history and
    its own temporal state, so a frame they produce exists only inside a runtime that had a swapchain.

So there are two halves, and the harness owns the seam between them:

    offline   arms driven from *our* frames - bilinear, bicubic, lanczos, FSR 1.0 (tools/fsr1.py),
              and the NRR model. Scored here, directly.
    online    arms that run in their own runtime and deposit what they produced - XeSS through its
              Vulkan sample, DLSS through the patched Godot fork. Each is a directory carrying
              `arm.json` (what it is, what resolution, what preset, what it cost) plus the frames and
              a reference when it produced its own.

Both halves are scored by the identical metric code, and every row carries a `same-frames` column. A
row that says `no` is a capability and latency measurement; a row that says `yes` is a quality
comparison. No sentence about quality spans two arms whose column says `no` - that is the rule this
file exists to enforce, because a table that mixes the two is how an arm's own sample scene becomes a
claim about our content.

Why the scoring loop is written out here instead of importing `compare_upscalers.py`'s: that tool's
report has no concept of an arm and no `same-frames` field, and its shape is what the recorded FSR 1.0
numbers were read from. Changing it would invalidate numbers already written down, so it stands and
this is the harness that has an arm concept.

Usage:

    python tools/parity_harness.py --self-test
    python tools/parity_harness.py --data models/training-data/godot-v6-warp --split val --limit 20 \
        --model models/phase4/upscale_msreal_scale.onnx \
        --online work/parity/arms/xess-vk-1080p \
        --out work/parity/report.json --markdown docs/parity.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as em    # noqa: E402  (loader, metrics, VMAF and perceptual plumbing)
import fsr1                     # noqa: E402  (the one open-source spatial upscaler ported in-tree)
import quality_metrics as qm    # noqa: E402

# The arms this harness can drive itself, in the order a table should read them: the fixed-function
# floor first, then the commercial spatial upscaler, then the model.
OFFLINE_METHODS = ("bilinear", "bicubic", "lanczos", "fsr1", "nrr")

# Keys an online arm must carry. `frames` and `reference` may be null, but only alongside
# `quality: unavailable` and a reason - a missing measurement is a stated fact, never a silent zero.
ARM_REQUIRED = ("arm", "version", "backend", "preset", "ratio", "output", "same_frames", "scene")

# A quality row is scored only if at least this many frames were available; below it the metric is
# reported unavailable with the count, because a mean over three frames is not a measurement.
MIN_FRAMES_FOR_QUALITY = 4


def load_arm(arm_dir):
    """One online arm as a dict, refused rather than defaulted when it is not usable.

    The point of requiring `arm.json` is that a number from a runtime we cannot re-run here is only
    worth recording if the thing that produced it is identified: version, backend, preset, ratio,
    resolutions. A directory of PNGs with no such file is a picture, not a measurement.
    """
    manifest_path = os.path.join(arm_dir, "arm.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("%s has no arm.json - an online arm must identify itself" % arm_dir)
    # utf-8-sig, not utf-8: arm.json is written by the PowerShell runner, whose -Encoding UTF8 emits a BOM,
    # and a BOM is not a reason to refuse a file whose contents are exactly right.
    with open(manifest_path, "r", encoding="utf-8-sig") as handle:
        arm = json.load(handle)
    missing = [key for key in ARM_REQUIRED if key not in arm]
    if missing:
        raise SystemExit("%s is missing %s" % (manifest_path, ", ".join(missing)))
    if not isinstance(arm["same_frames"], bool):
        raise SystemExit("%s: same_frames must be true or false, not %r" % (manifest_path, arm["same_frames"]))
    # An arm whose quality could not be measured must say why: the protocol's rule is that an
    # unavailable metric is reported with a reason, never silently passed.
    if arm.get("quality") == "unavailable" and not arm.get("quality_reason"):
        raise SystemExit("%s declares quality unavailable without a quality_reason" % manifest_path)
    arm["_dir"] = os.path.abspath(arm_dir)
    return arm


def frame_list(directory):
    """The frames in a directory as [(index, path)], numbered `frame_%04d.png`.

    A gap is refused rather than skipped: frame N is scored against the reference for frame N, so a
    missing frame in the middle would compare every later frame against the wrong reference and report
    a plausible-looking error instead of failing.
    """
    if not directory or not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("frame_") and name.lower().endswith(".png")):
            continue
        stem = os.path.splitext(name)[0][len("frame_"):]
        if not stem.isdigit():
            raise SystemExit("%s: %s is not frame_<number>.png" % (directory, name))
        found.append((int(stem), os.path.join(directory, name)))
    found.sort()
    for (before, _), (after, _) in zip(found, found[1:]):
        if after != before + 1:
            raise SystemExit("%s: frames jump from %d to %d - a gap misaligns every later reference"
                             % (directory, before, after))
    if found and found[0][0] != 0:
        raise SystemExit("%s: frames start at %d, expected 0" % (directory, found[0][0]))
    return found


def read_png(path):
    """A frame as float RGB in [0,1] - the same read every other tool in this tree uses."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def frame_metrics(upscaled, reference):
    """Raw per-frame metrics for one arm over one sequence.

    Kept raw rather than pre-averaged so several scenes can be pooled into one row without averaging
    averages - which would weight a 20-frame scene like a 200-frame one - while still reporting the
    spread. The protocol reports means *with* their sigma for exactly this reason.
    """
    keys = ("l1", "ssim", "ms_ssim", "psnr_db")
    values = {key: [] for key in keys}
    detail = []
    for up, target in zip(upscaled, reference):
        result = qm.evaluate(up, target)
        for key in keys:
            values[key].append(result[key])
        energy = em.high_frequency_energy(target)
        detail.append(em.high_frequency_energy(up) / energy if energy > 1e-9 else None)
    values["detail_ratio"] = detail
    return values


def summarise(raw):
    """The `accumulate` blocks for one arm, from raw per-frame values."""
    return {"image": {key: em.accumulate(raw[key]) for key in ("l1", "ssim", "ms_ssim", "psnr_db")},
            "detail_ratio": em.accumulate(raw["detail_ratio"])}


def merge_metrics(into, other):
    """Pool per-frame values from another scene into `into`, so one row covers every val scene."""
    for key, values in other.items():
        into.setdefault(key, []).extend(values)
    return into


def offline_upscalers(model_path, provider):
    """The arms this harness drives from our own frames: name -> callable(image, entry) -> upscaled image.

    Built by `tools/compare_upscalers.py`, deliberately: the FSR 1.0 row of the parity table is then
    produced by the same code and the same resample discipline as the FSR 1.0 numbers this project has
    already recorded, so the two cannot drift apart.

    The arms are given the scene entry as well as the image, because the exported model's own declared
    inputs decide its feed: a model trained with `--inputs=color,jitter` has to be told the phase its frame
    was sampled at, and that is a per-frame fact only the entry carries. Feeding it zeros would measure a
    different model's job while wearing this one's name.
    """
    import compare_upscalers as cu
    arms = cu.make_upscalers(model_path, provider)
    methods = {}
    for name, upscale in arms.items():
        if name == "nrr":
            arm = lambda image, entry, upscale=upscale: upscale(image, entry.get("jitter"))
        else:
            arm = lambda image, entry, upscale=upscale: upscale(image)
        # What the export declares is carried across the adapter, so the report can name the inputs behind
        # the row rather than the row arriving unattributed.
        declared = getattr(upscale, "declared_inputs", None)
        if declared:
            arm.declared_inputs = declared
        methods[name] = arm
    return methods


def evaluate_scene_offline(methods, entries, data_dir, perceptual, ffmpeg, workdir,
                           run_perceptual, run_vmaf):
    """One scene through every offline arm: per-arm metrics, plus perceptual and VMAF when available.

    VMAF needs the whole sequence, so the frames are kept as lists rather than scored frame by frame -
    the same shape `evaluate_model.py` and `compare_upscalers.py` use.
    """
    frames = {name: [] for name in methods}
    reference = []
    for entry in entries:
        pair = em.load_pair(data_dir, entry)
        reference.append(em.to_image(pair["target"]))
        for name, upscale in methods.items():
            frames[name].append(em.to_image(upscale(pair["input"], entry)))

    report = {}
    for name, produced in frames.items():
        entry = {"raw": frame_metrics(produced, reference)}
        if run_perceptual and perceptual is not None and (perceptual.lpips is not None
                                                          or perceptual.dists is not None):
            gathered = {"lpips": [], "dists": []}
            for up, target in zip(produced, reference):
                for key, value in perceptual.distances(up, target).items():
                    if value is not None:
                        gathered[key].append(value)
            entry["perceptual_raw"] = gathered
        if run_vmaf and ffmpeg is not None:
            entry["vmaf"] = em.vmaf_sequence(produced, reference, ffmpeg, workdir)
        report[name] = entry
    return report, reference


def evaluate_arm_online(arm, references, perceptual, ffmpeg, workdir, run_perceptual, run_vmaf):
    """One online arm's own frames, scored against its own reference (or ours when it shares them).

    `references` carries our scene's targets two ways: `pooled` (every val scene, in the order a shared-frame
    arm has to write its frames in) and `scene` (one parity scene). An arm that shares our frames is scored
    against whichever matches what it produced - the pool if it covers every scene, the single scene if it
    covers one - so an arm is never silently compared against 24 references while holding 48 frames. An arm
    running its own scene has its own truth, and scoring it against ours would measure the scene difference
    rather than the upscaler.
    """
    if arm.get("quality") == "unavailable":
        return {"quality": "unavailable", "reason": arm["quality_reason"]}
    produced_paths = frame_list(os.path.join(arm["_dir"], arm.get("frames", "frames")))
    if arm["same_frames"]:
        pooled = references.get("pooled", [])
        if len(produced_paths) == len(pooled) and pooled:
            reference = pooled
            reference_note = ("shared scene frames, pooled over %d val scene(s)"
                              % references.get("scene_count", 1))
        else:
            reference = references.get("scene", [])[:len(produced_paths)]
            reference_note = "shared scene frames (%s)" % references.get("scene_name", "?")
    else:
        reference_dir = os.path.join(arm["_dir"], arm.get("reference", "reference"))
        reference = [read_png(path) for _, path in frame_list(reference_dir)]
        reference_note = "the arm's own reference"
    if len(produced_paths) < MIN_FRAMES_FOR_QUALITY or len(reference) != len(produced_paths):
        return {"quality": "unavailable",
                "reason": ("%d frames produced against %d reference frames - below the %d-frame floor"
                           % (len(produced_paths), len(reference), MIN_FRAMES_FOR_QUALITY))}
    produced = [read_png(path) for _, path in produced_paths]
    if produced[0].shape != reference[0].shape:
        return {"quality": "unavailable",
                "reason": ("frame size %s against reference %s - not the same output resolution"
                           % (produced[0].shape, reference[0].shape))}
    # Summarised into the same shape an offline arm reports, deliberately: the markdown table, the console
    # summary and the JSON all read `image`/`detail_ratio`/`perceptual`, and an arm that arrives through a
    # different door should not need a second reader. (This is what the first shared-frame arm exposed: the
    # online path produced `raw` and the table's own printer then failed on a measured row.)
    entry = {"quality": "measured", "frames": len(produced), "reference": reference_note}
    entry.update(summarise(frame_metrics(produced, reference)))
    if run_perceptual and perceptual is not None and (perceptual.lpips is not None
                                                      or perceptual.dists is not None):
        gathered = {"lpips": [], "dists": []}
        for up, target in zip(produced, reference):
            for key, value in perceptual.distances(up, target).items():
                if value is not None:
                    gathered[key].append(value)
        entry["perceptual"] = {key: em.accumulate(values) for key, values in gathered.items()}
    if run_vmaf and ffmpeg is not None:
        sequence = em.vmaf_sequence(produced, reference, ffmpeg, workdir)
        mean = sequence.get("vmaf_mean") if isinstance(sequence, dict) else None
        if mean is not None:
            entry["vmaf"] = {"vmaf_mean": float(mean), "scenes": 1}
    return entry


# The C++ frame path's own breakdown line, from tests/performance/test_latency.cpp:
#   "  1920x1080: wall=1031.2ms  reported=989.6ms  (inference=157.4ms, host overhead=832.2ms)"
# It is parsed rather than re-measured here because it is a *different* measurement from
# measure_model.py: that one times an ONNX session, this one times the runtime's whole frame including the
# host-side data path. A parity table that mixed the two would report the smaller number as if it were the
# product's frame cost, so the type is a field on every latency row.
FRAME_BUDGET_LINE = re.compile(
    r"^\s*(?P<tier>\d+x\d+(?:->\d+x\d+)?):\s*wall=(?P<wall>[\d.]+)ms\s+reported=(?P<reported>[\d.]+)ms\s+"
    r"\(inference=(?P<inference>[\d.]+)ms,\s*host overhead=(?P<overhead>[\d.]+)ms\)"
    # Optional distribution, printed by nrr_bench and absent from the suite's own breakdown line. The tier
    # may also carry the output grid (`960x540->1920x1080`), which is how a tier is named once two arms have
    # to agree on it: XeSS's sample is told the output size and derives its input, so a bare "1080p" would
    # mean opposite ends in the two arms.
    r"(?:\s+p50=(?P<p50>[\d.]+)ms)?(?:\s+p99=(?P<p99>[\d.]+)ms)?(?:\s+fps=(?P<fps>[\d.]+))?")


def parse_frame_budget_log(text):
    """The per-tier frame breakdown out of a suite log, keyed by the tier name the test prints."""
    tiers = {}
    for line in text.splitlines():
        match = FRAME_BUDGET_LINE.match(line)
        if not match:
            continue
        numbers = {key: float(match.group(key)) for key in ("wall", "reported", "inference", "overhead")}
        entry = {
            "wall_ms": numbers["wall"], "reported_ms": numbers["reported"],
            "inference_ms": numbers["inference"], "host_overhead_ms": numbers["overhead"],
            "inference_share": (numbers["inference"] / numbers["reported"]) if numbers["reported"] else None,
        }
        # Present only in a benchmarker's line: the suite's breakdown prints an average and no distribution.
        for key in ("p50", "p99", "fps"):
            if match.group(key) is not None:
                entry[key + "_ms" if key != "fps" else "fps"] = float(match.group(key))
        tiers[match.group("tier")] = entry
    return tiers


def load_nrr_latency(model_path, frame_budget_path):
    """NRR's latency rows: what the session costs, and what the runtime's frame costs.

    The two are kept apart on purpose. `inference-only` comes from tools/measure_model.py's report beside
    the model; `end-to-end-frame` comes from the C++ breakdown test's own line, extracted by this file. If
    only one exists, only one row appears - an absent measurement is absent, not zero.
    """
    rows = []
    if model_path:
        latency_path = os.path.splitext(model_path)[0] + ".latency.json"
        if os.path.exists(latency_path):
            with open(latency_path, "r", encoding="utf-8-sig") as handle:
                report = json.load(handle)
            for tier, entry in report.get("tiers", {}).items():
                rows.append({"arm": "nrr", "tier": tier, "kind": "inference-only",
                             "ms": entry.get("ms_per_frame"),
                             "output": entry.get("output"),
                             "source": "tools/measure_model.py, providers=%s"
                                       % ",".join(report.get("active_providers", []) or [])})
    if frame_budget_path and os.path.exists(frame_budget_path):
        with open(frame_budget_path, "r", encoding="utf-8-sig") as handle:
            tiers = json.load(handle)
        for tier, entry in tiers.items():
            rows.append({"arm": "nrr", "tier": tier, "kind": "end-to-end-frame",
                         "ms": entry.get("reported_ms"),
                         "output": None,
                         "source": ("latency_frame_budget_breakdown: inference %s ms + host overhead %s ms"
                                    % (entry.get("inference_ms"), entry.get("host_overhead_ms")))})
    return rows


def load_nrr_bench(path, model_path=""):
    """NRR's `end-to-end-frame` rows from `benchmarks/nrr_bench.cpp`'s own report.

    This is the arm's counterpart to XeSS's sample benchmark: a standalone binary, a chosen tier, a chosen
    frame count, and a distribution rather than an average. The runtime's debug string travels with the row,
    because it is what makes the number interpretable - it names the execution provider that actually
    attached and whether the temporal blend engaged (`temporal alpha=0 (motion above threshold)` is a
    different measurement from one where it did).

    A report older than the model it names is refused rather than published: after a retrain, the previous
    run's rows describe weights that no longer exist, and nothing in the row itself would say so.
    """
    rows = []
    if not path or not os.path.exists(path):
        return rows
    if model_path and os.path.exists(model_path) and os.path.getmtime(path) < os.path.getmtime(model_path):
        em.log("ignoring %s: it is older than %s, so its rows describe weights that were replaced"
               % (os.path.basename(path), os.path.basename(model_path)))
        return rows
    with open(path, "r", encoding="utf-8-sig") as handle:
        report = json.load(handle)
    model = os.path.basename(report.get("model", "?"))
    for tier, entry in sorted(report.get("tiers", {}).items()):
        wall = entry.get("wall_ms")
        fps = entry.get("fps")
        source = ("nrr_bench, %s, %s frames on %s" %
                  (model, entry.get("frames", "?"), entry.get("provider", "unknown")))
        if entry.get("p50_ms") is not None and entry.get("p99_ms") is not None:
            source += ", p50 %.2f ms p99 %.2f ms" % (entry["p50_ms"], entry["p99_ms"])
        if wall is not None:
            source += ", wall %.2f ms" % wall
        if fps is not None:
            source += ", %.1f fps" % fps
        rows.append({"arm": "nrr", "tier": tier, "kind": "end-to-end-frame",
                     "ms": entry.get("reported_ms"),
                     "output": entry.get("output"),
                     "model": model,
                     "runtime_debug": entry.get("runtime_debug", ""),
                     "source": source})
    return rows


def arm_latency_rows(arm):
    """The latency rows an online arm declares, as a list so an arm may report several tiers."""
    declared = arm.get("latency_ms")
    if not declared:
        return [{"arm": arm["arm"], "tier": arm.get("latency_tier", "-"), "kind": "end-to-end-frame",
                 "ms": None, "output": arm.get("output"),
                 "source": arm.get("latency_reason", "not measured")}]
    if isinstance(declared, dict):
        declared = [declared]
    rows = []
    for entry in declared:
        rows.append({"arm": arm["arm"], "tier": entry.get("tier", arm.get("latency_tier", "-")),
                     "kind": entry.get("kind", "end-to-end-frame"), "ms": entry.get("ms"),
                     "output": entry.get("output", arm.get("output")),
                     "source": entry.get("source", arm.get("notes", ""))})
    return rows


def self_test():
    """The checks that need no model, dataset, torch, ffmpeg or GPU.

    Each is here because it is a way this harness could silently produce a wrong table: a misparsed
    breakdown line, a frame gap that misaligns references, an arm accepted without identifying itself, a
    latency row invented for an arm that never reported one, and a quality cell filled in for an arm
    whose `same-frames` says the comparison is not valid.
    """
    failures = []
    total = 0

    def check(name, condition, detail=""):
        nonlocal total
        total += 1
        if not condition:
            failures.append("%s%s" % (name, (": " + detail) if detail else ""))

    log = ("--- Latency Tests ---\n"
           "  256x256: wall=32.9ms  reported=32.7ms  (inference=3.7ms, host overhead=29.0ms)\n"
           "  1920x1080: wall=1031.2ms  reported=989.6ms  (inference=157.4ms, host overhead=832.2ms)\n"
           "  some other line\n")
    tiers = parse_frame_budget_log(log)
    check("parse_frame_budget_log finds both tiers", set(tiers) == {"256x256", "1920x1080"}, str(set(tiers)))
    check("parse_frame_budget_log reads the reported total",
          tiers.get("1920x1080", {}).get("reported_ms") == 989.6, str(tiers.get("1920x1080")))
    check("parse_frame_budget_log computes the inference share",
          abs(tiers.get("256x256", {}).get("inference_share", 0) - (3.7 / 32.7)) < 1e-9)
    check("parse_frame_budget_log ignores prose", parse_frame_budget_log("no tiers here") == {})

    # The benchmarker's line: the same shape the suite prints, plus the output grid and the distribution. The
    # tier name has to survive parsing intact, because that is the name the two arms' rows are compared by.
    bench_log = ("  960x540->1920x1080: wall=110.07ms  reported=106.73ms  "
                 "(inference=53.66ms, host overhead=53.07ms)  p50=107.69ms  p99=133.10ms  fps=9.1\n")
    bench_tiers = parse_frame_budget_log(bench_log)
    check("parse_frame_budget_log keeps the input->output tier name",
          set(bench_tiers) == {"960x540->1920x1080"}, str(set(bench_tiers)))
    check("parse_frame_budget_log reads the distribution when there is one",
          bench_tiers["960x540->1920x1080"]["p99_ms"] == 133.10
          and bench_tiers["960x540->1920x1080"]["fps"] == 9.1, str(bench_tiers))
    check("a suite line without a distribution still parses",
          "p99_ms" not in tiers["256x256"] and "fps" not in tiers["256x256"])

    with tempfile.TemporaryDirectory(prefix="nrr-parity-selftest-") as work:
        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir)
        for index in (0, 1, 2):
            open(os.path.join(frames_dir, "frame_%04d.png" % index), "wb").close()
        check("frame_list accepts a contiguous run", len(frame_list(frames_dir)) == 3)
        open(os.path.join(frames_dir, "frame_0004.png"), "wb").close()
        try:
            frame_list(frames_dir)
            check("frame_list refuses a gap", False, "it accepted frame 0004 after 0002")
        except SystemExit as error:
            check("frame_list refuses a gap", "jump" in str(error))

        arm_dir = os.path.join(work, "good-arm")
        os.makedirs(arm_dir)
        good = {"arm": "xess", "version": "3.0.2", "backend": "vulkan-dp4a", "preset": "performance",
                "ratio": 2.0, "output": [1920, 1080], "same_frames": False, "scene": "sample",
                "quality": "unavailable", "quality_reason": "the sample cannot dump frames"}
        with open(os.path.join(arm_dir, "arm.json"), "w", encoding="utf-8") as handle:
            json.dump(good, handle)
        arm = load_arm(arm_dir)
        check("load_arm accepts an identified arm", arm["arm"] == "xess")
        rows = arm_latency_rows(arm)
        check("an arm with no latency gets one row carrying the reason, not a zero",
              len(rows) == 1 and rows[0]["ms"] is None and "not measured" in rows[0]["source"], str(rows))
        arm_with_latency = dict(arm, latency_ms=[{"tier": "540p->1080p", "ms": 0.707},
                                                 {"tier": "1080p->4K", "ms": 2.01}])
        check("an arm with two tiers gets two rows", len(arm_latency_rows(arm_with_latency)) == 2)

        # The runner writes arm.json through PowerShell, whose `-Encoding UTF8` emits a BOM. A BOM must not
        # be the reason a correct file is refused, or the whole runner path fails at ingestion.
        bom_dir = os.path.join(work, "bom-arm")
        os.makedirs(bom_dir, exist_ok=True)
        with open(os.path.join(bom_dir, "arm.json"), "w", encoding="utf-8-sig") as handle:
            json.dump(good, handle)
        check("load_arm reads the BOM-prefixed arm.json that PowerShell writes",
              load_arm(bom_dir)["arm"] == "xess")

        # The benchmarker's report: a row per tier, carrying what the runtime said about the frame it measured.
        # The debug string is the reason the row is interpretable - it names the provider that attached and
        # whether the temporal blend engaged - so losing it would be losing the measurement's conditions.
        bench_path = os.path.join(work, "nrr-bench.json")
        with open(bench_path, "w", encoding="utf-8-sig") as handle:
            json.dump({"tool": "nrr_bench", "model": "C:\\models\\upscale_subsampled.onnx", "ratio": 2.0,
                       "frames": 20, "warmup": 3,
                       "tiers": {"960x540->1920x1080": {
                           "input": [960, 540], "output": [1920, 1080], "frames": 20,
                           "reported_ms": 89.46, "inference_ms": 46.72, "host_overhead_ms": 42.75,
                           "wall_ms": 91.84, "p50_ms": 90.98, "p99_ms": 106.96, "fps": 10.9,
                           "provider": "CUDAExecutionProvider",
                           "runtime_debug": "ONNX CUDAExecutionProvider via accel 960x540 -> 1920x1080"}}},
                      handle)
        bench_rows = load_nrr_bench(bench_path)
        check("load_nrr_bench makes one end-to-end row per tier",
              len(bench_rows) == 1 and bench_rows[0]["kind"] == "end-to-end-frame"
              and bench_rows[0]["tier"] == "960x540->1920x1080", str(bench_rows))
        check("a bench row publishes the runtime's own reported time, not the wall clock",
              bench_rows[0]["ms"] == 89.46 and "wall 91.84 ms" in bench_rows[0]["source"], str(bench_rows[0]))
        check("a bench row carries the provider that actually attached and its distribution",
              bench_rows[0]["runtime_debug"].startswith("ONNX CUDAExecutionProvider")
              and "p99 106.96 ms" in bench_rows[0]["source"], str(bench_rows[0]["source"]))
        check("load_nrr_bench on a missing file makes no rows", load_nrr_bench("") == [])

        # A report from before a retrain describes weights that no longer exist, and nothing in the row would
        # say so: it is refused rather than published.
        stale = os.path.join(work, "stale-bench.json")
        with open(stale, "w", encoding="utf-8") as handle:
            json.dump({"tool": "nrr_bench", "model": "old.onnx", "tiers": {}}, handle)
        newer_model = os.path.join(work, "newer.onnx")
        with open(newer_model, "wb") as handle:
            handle.write(b"onnx")
        os.utime(newer_model, (os.path.getmtime(stale) + 60, os.path.getmtime(stale) + 60))
        check("load_nrr_bench refuses a report older than the model it measured",
              load_nrr_bench(stale, newer_model) == [] and load_nrr_bench(stale) == [])

        for reason, broken in (("a missing key", dict(good, backend=None)),
                               ("unavailable without a reason", dict(good, quality_reason=None)),
                               ("same_frames that is not a bool", dict(good, same_frames="yes"))):
            sub = os.path.join(work, "broken-" + reason.replace(" ", "-"))
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, "arm.json"), "w", encoding="utf-8") as handle:
                json.dump({key: value for key, value in broken.items() if value is not None}, handle)
            try:
                load_arm(sub)
                check("load_arm refuses %s" % reason, False, "it accepted it")
            except SystemExit:
                check("load_arm refuses %s" % reason, True)

    rng = np.random.default_rng(20261009)
    frame = rng.random((8, 8, 3)).astype(np.float32)
    raw = frame_metrics([frame], [frame])
    check("an identical frame scores L1 0 and detail 1.0",
          raw["l1"] == [0.0] and abs(raw["detail_ratio"][0] - 1.0) < 1e-6, str(raw))
    flat = frame_metrics([np.zeros((8, 8, 3), np.float32)], [np.zeros((8, 8, 3), np.float32)])
    check("a flat reference yields no detail ratio rather than a fabricated 1.0",
          flat["detail_ratio"] == [None], str(flat))
    pooled = merge_metrics(merge_metrics({}, raw), raw)
    check("merge_metrics pools frames rather than averaging averages", len(pooled["l1"]) == 2)

    # The feed contract the nrr row depends on: a model that declares jitter must be given the frame's own
    # offset, broadcast to a plane, and must refuse to run with nothing rather than substituting zeros.
    import compare_upscalers as cu
    feed = cu.build_feed_from_image(frame, ["color", "jitter"], jitter=[0.25, -0.5])
    check("a jitter-aware feed broadcasts the frame's offset to a plane",
          feed["jitter"].shape == (1, 2, 8, 8) and abs(feed["jitter"][0, 0].mean() - 0.25) < 1e-6
          and abs(feed["jitter"][0, 1].mean() + 0.5) < 1e-6, str(feed["jitter"].shape))
    try:
        cu.build_feed_from_image(frame, ["color", "jitter"])
        check("a jitter-aware feed refuses to invent a phase", False, "it accepted a missing offset")
    except ValueError:
        check("a jitter-aware feed refuses to invent a phase", True)
    check("an image-only feed still gets zeros for its other declared inputs",
          set(cu.build_feed_from_image(frame, ["color", "depth", "motion"])) == {"color", "depth", "motion"})
    # The resolution token is derived from the image, so a model that declares it is scored at whatever tier the
    # dataset provides without the caller having to know: 128 -> 0.0, 256 -> 1.0.
    token = cu.build_feed_from_image(np.zeros((256, 256, 3), np.float32), ["color", "scale"])["scale"]
    check("the scale token is derived from the image's own width",
          token.shape == (1, 1, 256, 256) and abs(float(token.mean()) - 1.0) < 1e-6, str(token.shape))
    small = cu.build_feed_from_image(np.zeros((128, 128, 3), np.float32), ["color", "scale"])["scale"]
    check("the scale token is 0.0 at the reference tier", abs(float(small.mean())) < 1e-6)

    report = {
        "scene": {"name": "temporal", "ratio": 2.0}, "data": "d", "split": "val", "scene_frames": 10,
        "machine": {"gpu": "RTX 4070 Ti", "driver": "610.88"},
        "arms": {
            "bilinear": {"kind": "offline", "same_frames": True,
                         "identity": {"version": "torch", "backend": "PyTorch", "preset": "2x",
                                      "tier": "128x128->256x256"},
                         "quality": {"quality": "measured", "frames": 10,
                                     "image": {key: {"mean": 1.0, "n": 10} for key in
                                               ("psnr_db", "ssim", "ms_ssim")},
                                     "detail_ratio": {"mean": 0.5, "n": 10}}},
            "xess": {"kind": "online", "same_frames": False,
                     "identity": {"version": "3.0.2", "backend": "vulkan", "preset": "performance",
                                  "tier": "128x128->256x256"},
                     "quality": {"quality": "unavailable", "reason": "no frame dump"}},
        },
        "latency": [{"arm": "xess", "tier": "540p->1080p", "kind": "end-to-end-frame", "ms": 0.707,
                     "output": [1920, 1080], "source": "sample benchmark"}],
        "unavailable": [{"arm": "xess", "what": "quality", "reason": "no frame dump"}],
    }
    table = render_markdown(report)
    check("the table marks the own-scene arm as not-same-frames", "| no |" in table)
    check("the table marks the shared-frame arm as same-frames", "| yes |" in table)
    check("every row names the tier that produced it", table.count("128x128->256x256") >= 2, table[:200])
    check("an own-scene arm's quality cells are empty rather than filled",
          "| xess | 128x128->256x256 | 3.0.2 | vulkan | performance | no | - | - | - | - | - | - | - | - |" in table)
    check("the latency row keeps its measurement kind",
          "| xess | 540p->1080p | end-to-end-frame | 0.707 |" in table)

    print("RESULT: %s (%d checks)" % ("PASS" if not failures else "FAIL", total))
    for failure in failures:
        print("  FAILED: %s" % failure)
    return 1 if failures else 0


def machine_info():
    """The GPU and driver the latency numbers belong to, read from the driver rather than guessed."""
    info = {"gpu": "unknown", "driver": "unknown"}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20)
        if result.returncode == 0 and result.stdout.strip():
            first = result.stdout.strip().splitlines()[0]
            name, _, driver = first.partition(",")
            info["gpu"], info["driver"] = name.strip(), driver.strip()
    except (OSError, subprocess.SubprocessError):
        # A machine without nvidia-smi still produces a table; it just says the GPU is unknown rather
        # than naming one it did not check.
        pass
    return info


def build_parser():
    parser = argparse.ArgumentParser(description="The DLSS/XeSS parity table, produced by one command.")
    parser.add_argument("--data", default="models/training-data/godot-v6-warp",
                        help="the packed dataset whose frames the offline arms and shared-frame arms use")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0, help="cap frames per scene (0 = all)")
    parser.add_argument("--scene", default="", help="the scene an online arm's frames correspond to")
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--model", default=None, help="the exported .onnx for the nrr arm")
    parser.add_argument("--provider", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--online", action="append", default=[],
                        help="a directory carrying arm.json from a runtime we cannot re-run here "
                             "(repeatable)")
    parser.add_argument("--frame-budget-log", default="",
                        help="a suite log to extract the runtime's per-tier frame breakdown from")
    parser.add_argument("--frame-budget-out", default="",
                        help="where to write the extracted breakdown as JSON")
    parser.add_argument("--nrr-frame-budget", default="",
                        help="a previously extracted breakdown JSON to read the end-to-end rows from")
    parser.add_argument("--nrr-bench", default="",
                        help="benchmarks/nrr_bench.cpp's JSON report, the arm's own end-to-end measurement")
    parser.add_argument("--skip-perceptual", action="store_true")
    parser.add_argument("--skip-vmaf", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="work/parity/report.json")
    parser.add_argument("--markdown", default="", help="write the generated table here (e.g. docs/parity.md)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the parsing, refusal and rendering checks and exit (no model, dataset, "
                             "torch, ffmpeg or GPU needed - this is what CI runs)")
    return parser


# ---------------------------------------------------------------------------
# The table (continued)
# ---------------------------------------------------------------------------

# What each arm *is*, so the quality table can identify a row the way the ARM_REQUIRED keys make an
# online arm identify itself. `same_frames` is true for every offline arm by construction: they are
# driven from the scene's own low-resolution frames and scored against the scene's own targets.
OFFLINE_IDENTITY = {
    "bilinear": ("torch interpolate, align_corners=False", "PyTorch", "2x"),
    "bicubic": ("Pillow BICUBIC on 8-bit", "CPU, fixed-function", "2x"),
    "lanczos": ("Pillow LANCZOS on 8-bit", "CPU, fixed-function", "2x"),
    "fsr1": ("FSR 1.0 (EASU + RCAS)", "CPU port, tools/fsr1.py", "2x"),
    "nrr": ("nrr_upscaler_trained", "ONNX Runtime", "2x"),
}


def _stat(stats, key="mean", digits=4):
    """One number out of an `accumulate` block, or `-` when the metric was not available."""
    if not stats:
        return "-"
    value = stats.get(key)
    return "-" if value is None else ("%.*f" % (digits, value))


def render_markdown(report):
    """`docs/parity.md` from the report. Generated, never hand-edited."""
    lines = []
    scene = report["scene"]
    lines.append("# Parity: the head-to-head table")
    lines.append("")
    lines.append("Generated by `tools/parity_harness.py` - **do not edit by hand**. Regenerate with:")
    lines.append("")
    lines.append("```powershell")
    lines.append("powershell -File tools\\run_parity.ps1 -Nrr -Xess -Dlss")
    lines.append("```")
    lines.append("")
    lines.append("Scene: **%s** (`%s`, %s split, %d frames, %.2fx). GPU: %s, driver %s. "
                 "A timing taken on a host doing other work is a timing on a busy host."
                 % (scene.get("name", "?"), report["data"], report["split"], report["scene_frames"],
                    scene.get("ratio", 2.0), report["machine"].get("gpu", "?"),
                    report["machine"].get("driver", "?")))
    lines.append("")
    lines.append("`same-frames` is the column that decides what a row may be used for. **yes** means the arm")
    lines.append("was driven from this scene's own frames and scored against this scene's own targets, so its")
    lines.append("row is a quality comparison. **no** means the arm ran in its own runtime on its own content,")
    lines.append("so its row is a capability and latency measurement and its quality columns are empty by")
    lines.append("construction. No sentence about quality may span a mixed pair.")
    lines.append("")
    lines.append("## Quality")
    lines.append("")
    lines.append("| arm | tier | version | backend | preset | same-frames | frames | PSNR dB | SSIM | MS-SSIM | "
                 "LPIPS | DISTS | VMAF | detail |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, arm in report["arms"].items():
        quality = arm.get("quality") or {}
        identity = arm.get("identity", {})
        if quality.get("quality") != "measured":
            lines.append("| %s | %s | %s | %s | %s | %s | - | - | - | - | - | - | - | - |"
                         % (name, identity.get("tier", "-"), identity.get("version", "?"),
                            identity.get("backend", "?"), identity.get("preset", "?"),
                            "yes" if arm.get("same_frames") else "no"))
            continue
        perceptual = quality.get("perceptual") or {}
        vmaf = (quality.get("vmaf") or {}).get("vmaf_mean")
        lines.append("| %s | %s | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |"
                     % (name, identity.get("tier", "-"), identity.get("version", "?"),
                        identity.get("backend", "?"), identity.get("preset", "?"),
                        "yes" if arm["same_frames"] else "no",
                        quality.get("frames", report["scene_frames"]),
                        _stat(quality["image"]["psnr_db"]), _stat(quality["image"]["ssim"], digits=5),
                        _stat(quality["image"]["ms_ssim"], digits=5),
                        _stat(perceptual.get("lpips")), _stat(perceptual.get("dists")),
                        "-" if vmaf is None else "%.3f" % vmaf,
                        _stat(quality["detail_ratio"])))
    lines.append("")
    lines.append("The offline arms are driven from the `input` plane the scene's own frames carry; `nrr` is the")
    lines.append("named model, fed exactly what the trainer exports.")
    lines.append("")
    declared = [(name, arm["identity"]["inputs"]) for name, arm in report["arms"].items()
                if (arm.get("identity") or {}).get("inputs")]
    if declared:
        lines.append("A model's row names the inputs its export declares, because that is what makes the row")
        lines.append("interpretable - the same weights fed a different input set are a different arm:")
        lines.append("")
        for name, inputs in sorted(declared):
            lines.append("* `%s` = `%s`" % (name, inputs))
        lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append("`inference-only` is an ONNX session's cost. `end-to-end-frame` is the runtime's whole frame,")
    lines.append("host data path included. The two are not comparable and are never averaged together.")
    lines.append("")
    lines.append("| arm | tier | measurement | ms | output | source |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in report["latency"]:
        output = row.get("output")
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (row["arm"], row["tier"], row["kind"],
                        "-" if row["ms"] is None else "%.3f" % row["ms"],
                        "-" if not output else "%dx%d" % (output[0], output[1]), row["source"]))
    lines.append("")
    debugged = [row for row in report["latency"] if row.get("runtime_debug")]
    if debugged:
        lines.append("What the runtime said about its own frame, verbatim, for the rows it measured - the")
        lines.append("provider that actually attached and whether the temporal blend engaged are part of the")
        lines.append("measurement, not annotations on it:")
        lines.append("")
        for row in debugged:
            lines.append("* `%s` at **%s**: `%s`" % (row["arm"], row["tier"], row["runtime_debug"]))
        lines.append("")
    lines.append("## Not measured, and why")
    lines.append("")
    lines.append("| arm | what | reason |")
    lines.append("| --- | --- | --- |")
    for note in report["unavailable"]:
        lines.append("| %s | %s | %s |" % (note["arm"], note["what"], note["reason"]))
    if not report["unavailable"]:
        lines.append("| - | - | nothing outstanding |")
    lines.append("")
    return "\n".join(lines)


def pool_scenes(scenes, methods, data_dir, perceptual, ffmpeg, workdir, run_perceptual, run_vmaf):
    """Every val scene through every offline arm, pooled into one raw-metric set per arm.

    Pooled rather than averaged per scene so a 20-frame scene cannot weigh as much as a 200-frame one,
    and so the spread in the table is over frames rather than over scenes. The references come back too,
    because an arm that shares our frames is scored against this scene's own targets.
    """
    pooled, perceptual_pooled, vmaf_by_arm, references = {}, {}, {}, {}
    frames_total = 0
    for scene, entries in sorted(scenes.items()):
        if len(entries) < 2:
            em.log("scene %s: %d frame(s), not a sequence - skipped" % (scene, len(entries)))
            continue
        em.log("scene %s (%d frames)" % (scene, len(entries)))
        report, reference = evaluate_scene_offline(methods, entries, data_dir, perceptual, ffmpeg,
                                                   workdir, run_perceptual, run_vmaf)
        references[scene] = reference
        frames_total += len(entries)
        for name, entry in report.items():
            merge_metrics(pooled.setdefault(name, {}), entry["raw"])
            if entry.get("perceptual_raw"):
                bucket = perceptual_pooled.setdefault(name, {"lpips": [], "dists": []})
                for key, values in entry["perceptual_raw"].items():
                    bucket[key].extend(values)
            if entry.get("vmaf"):
                vmaf_by_arm.setdefault(name, []).append(entry["vmaf"].get("vmaf_mean"))
    return pooled, perceptual_pooled, vmaf_by_arm, references, frames_total


def parse_args_and_extract(args):
    """The `--frame-budget-log` step, which is its own mode and returns None when it handled the run.

    Kept separate because mixing a log-parsing mode into the scoring path would make "which measurement
    is this row" depend on which flags happened to be set.
    """
    if not args.frame_budget_log:
        return False
    if not os.path.exists(args.frame_budget_log):
        raise SystemExit("no such log: %s" % args.frame_budget_log)
    with open(args.frame_budget_log, "r", encoding="utf-8", errors="replace") as handle:
        tiers = parse_frame_budget_log(handle.read())
    if not tiers:
        raise SystemExit("%s carries no '<tier>: wall=... reported=... (inference=..., host overhead=...) "
                         "line, so the breakdown test did not run in that log - nothing is extracted "
                         "rather than an empty table being written" % args.frame_budget_log)
    out_path = args.frame_budget_out or "work/parity/nrr-frame-budget.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(tiers, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for tier, entry in sorted(tiers.items()):
        print("  %-10s reported %9.3f ms (inference %8.3f, host overhead %9.3f)"
              % (tier, entry["reported_ms"], entry["inference_ms"], entry["host_overhead_ms"]))
    print("frame budget extracted from %s -> %s" % (args.frame_budget_log, out_path))
    return True


def build_offline_arms(methods, args, pooled, perceptual_pooled, vmaf_by_arm, scene_tier="-"):
    """One table row per offline arm, from the pooled raw metrics."""
    arms = {}
    for name in methods:
        if name not in pooled:
            continue
        quality = {"quality": "measured", "frames": len(pooled[name]["l1"])}
        quality.update(summarise(pooled[name]))
        if perceptual_pooled.get(name):
            quality["perceptual"] = {key: em.accumulate(values)
                                     for key, values in perceptual_pooled[name].items()}
        if vmaf_by_arm.get(name):
            quality["vmaf"] = {"vmaf_mean": float(np.mean(vmaf_by_arm[name])),
                               "scenes": len(vmaf_by_arm[name])}
        version, backend, preset = OFFLINE_IDENTITY.get(name, ("?", "?", "%.0fx" % args.ratio))
        # The tier is a column, not a header: two datasets at different resolutions produce two tables, and a
        # reader must be able to see from the row itself which grid produced the number.
        identity = {"version": version, "backend": backend, "preset": preset, "tier": scene_tier}
        if name == "nrr" and args.model:
            identity["version"] = os.path.basename(args.model)
            # The declared inputs travel into the report: a row produced by a model fed color,jitter has to
            # say so, or its number is not the number the reader thinks it is.
            declared = getattr(methods[name], "declared_inputs", None)
            if declared:
                identity["inputs"] = ",".join(declared)
        arms[name] = {"kind": "offline", "same_frames": True, "quality": quality, "identity": identity}
    return arms


def main(argv):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if parse_args_and_extract(args):
        return 0

    manifest = em.load_manifest(args.data)
    scenes = em.group_sequences(manifest, args.split, args.limit)
    if not scenes:
        raise SystemExit("no %s sequences in %s" % (args.split, args.data))

    # The tier every offline arm ran at, taken from the packed dataset's own record of its frames rather than
    # from the file name: `size` is the input grid, and a 2x pack makes the target twice it.
    side = int(manifest.get("size") or 0)
    scene_tier = "%dx%d->%dx%d" % (side, side, side * 2, side * 2) if side else "-"

    methods = offline_upscalers(args.model, args.provider)
    if not args.model:
        # Without a model there is no nrr arm: an all-zeros feed would produce a row for a network that
        # does not exist, which is the kind of fabricated capability this project refuses.
        methods.pop("nrr", None)
        em.log("no --model given, so the table carries no nrr row")

    run_perceptual = not args.skip_perceptual
    perceptual_device = "cpu"
    if args.device == "cuda":
        perceptual_device = "cuda"
    elif args.device == "auto" and em.torch is not None and em.torch.cuda.is_available():
        perceptual_device = "cuda"
    perceptual = em.Perceptual(device=perceptual_device) if run_perceptual else None
    ffmpeg, has_libvmaf, has_vmafmotion = em.ffmpeg_probe()
    run_vmaf = not args.skip_vmaf and has_libvmaf

    unavailable = []
    if not has_libvmaf and not args.skip_vmaf:
        unavailable.append({"arm": "(every arm)", "what": "VMAF",
                            "reason": "ffmpeg is missing or was built without libvmaf"})
    if not run_perceptual:
        unavailable.append({"arm": "(every arm)", "what": "LPIPS/DISTS", "reason": "disabled by flag"})

    with tempfile.TemporaryDirectory(prefix="nrr-parity-") as workdir:
        pooled, perceptual_pooled, vmaf_by_arm, references, frames_total = pool_scenes(
            scenes, methods, args.data, perceptual, ffmpeg, workdir, run_perceptual, run_vmaf)
        if not pooled:
            raise SystemExit("no scene in %s produced a scored frame" % args.data)

        arms = build_offline_arms(methods, args, pooled, perceptual_pooled, vmaf_by_arm, scene_tier)

        parity_scene = args.scene or sorted(references)[0]
        reference_frames = references.get(parity_scene, [])
        # The pooled reference list, in the order the exporter of a shared-frame arm has to write its frames in:
        # scene by scene, sorted, frames in capture order. An arm covering every val scene is scored against it;
        # an arm covering one scene is scored against that scene's frames. Which one it is, is decided by what
        # the arm actually produced rather than by asking it, and the row records which reference was used.
        pooled_reference = [frame for scene in sorted(references) for frame in references[scene]]
        arm_references = {"pooled": pooled_reference, "scene": reference_frames, "scene_name": parity_scene,
                          "scene_count": len(references)}
        for arm_dir in args.online:
            arm = load_arm(arm_dir)
            if arm["same_frames"] and not reference_frames:
                raise SystemExit("%s shares this scene's frames, but scene %r has no reference frames"
                                 % (arm_dir, parity_scene))
            quality = evaluate_arm_online(arm, arm_references, perceptual, ffmpeg, workdir,
                                          run_perceptual, run_vmaf)
            if quality.get("quality") != "measured":
                unavailable.append({"arm": arm["arm"], "what": "quality",
                                    "reason": quality.get("reason", "not measured")})
            arms[arm["arm"]] = {"kind": "online", "same_frames": arm["same_frames"], "quality": quality,
                                "identity": {"version": arm["version"], "backend": arm["backend"],
                                             "preset": arm["preset"],
                                             "tier": ("%dx%d->%dx%d" % tuple(arm["input"] + arm["output"]))
                                             if arm.get("input") and arm.get("output") else scene_tier}}

    latency = load_nrr_latency(args.model, args.nrr_frame_budget) if args.model else []
    # The benchmarker's own rows come after the session and suite rows: same kind, better provenance (it names
    # the provider that attached and the frames it measured), and it is the arm's counterpart to the XeSS
    # sample's benchmark rather than a by-product of the test suite.
    latency += load_nrr_bench(args.nrr_bench, args.model)
    for arm_dir in args.online:
        rows = arm_latency_rows(load_arm(arm_dir))
        latency.extend(rows)
        for row in rows:
            if row["ms"] is None:
                unavailable.append({"arm": row["arm"], "what": "latency", "reason": row["source"]})

    report = {
        "generated_by": "tools/parity_harness.py",
        "data": args.data, "split": args.split, "limit": args.limit, "ratio": args.ratio,
        "scene": {"name": args.scene or "all val scenes", "ratio": args.ratio},
        "scene_frames": frames_total,
        "scenes": sorted(scenes),
        "machine": machine_info(),
        "model": args.model,
        "capabilities": {"perceptual": bool(perceptual and (perceptual.lpips is not None
                                                            or perceptual.dists is not None)),
                         "vmaf": has_libvmaf, "vmafmotion": has_vmafmotion},
        "arms": arms, "latency": latency, "unavailable": unavailable,
    }
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("")
    print("parity: %d arm(s), %d frame(s) over %d scene(s) - %s, driver %s"
          % (len(arms), frames_total, len(report["scenes"]), report["machine"]["gpu"],
             report["machine"]["driver"]))
    for name, arm in arms.items():
        quality = arm.get("quality") or {}
        if quality.get("quality") == "measured":
            print("  %-10s same-frames=%s  psnr %.2f dB  ssim %.4f  ms-ssim %.4f  detail %.3f  (%d frames)"
                  % (name, "yes" if arm["same_frames"] else "no",
                     quality["image"]["psnr_db"]["mean"], quality["image"]["ssim"]["mean"],
                     quality["image"]["ms_ssim"]["mean"], quality["detail_ratio"]["mean"],
                     quality["frames"]))
        else:
            print("  %-10s same-frames=%s  quality unavailable: %s"
                  % (name, "yes" if arm["same_frames"] else "no", quality.get("reason", "")))
    print("report: %s" % out_path)

    if args.markdown:
        markdown = render_markdown(report)
        parent = os.path.dirname(os.path.abspath(args.markdown))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown)
        print("table: %s" % args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

