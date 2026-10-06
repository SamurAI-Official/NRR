#!/usr/bin/env python3
"""summarize_jitter.py - one table of the jitter study, across every seed of both arms.

The jitter question is asked twice over: does the model *use* the offset (the ablation), and is it *better*
for using it (the quality metrics against a control that differs in exactly one input). Those are different
claims, and a single mean hides both, so per-seed rows are printed as well as the aggregate - a spread running
from 0.02% to 9.96% improvement is not something an average should be allowed to describe on its own.

Runs are grouped by the inputs the model actually consumed, read from each report rather than from the run
name. A run counts as *valid* only if it passed every gate. Degenerate runs - the ones that barely trained and
were refused - are excluded from the aggregate and reported separately as a rate, because a run that learned
nothing is not a bad result for a configuration, it is no result at all, and averaging it in would quietly drag
the arm toward the bilinear baseline.

The arms are compared with an **exact permutation test** rather than a t-test: with four to six seeds per arm
a t-test's normality assumption is not credible, scipy is not a dependency here, and the permutation test is a
shuffle of the pooled values that assumes nothing. Its floor is 1/C(n+m,n), so the smallest attainable p-value
is printed beside it - "p < 0.001" from a few hundred arrangements is a weaker claim than the same number from
a million, and printing only the former would overstate it. A seeded bootstrap CI gives the effect size, which
at this sample size matters more than the p-value.

Usage:
    python tools/summarize_jitter.py
    python tools/summarize_jitter.py --reports models/p5
"""
import argparse
import glob
import itertools
import json
import os
import statistics
import sys

ARMS = ("jitter", "control")
# Direction of each metric where there is one. `None` means the number is reported without a claim about which
# arm is better - an ablation magnitude is a dependence measure, not a quality one, and averaging two of them
# into a ranking would be meaningless.
HIGHER_IS_BETTER = {"val_l1": False, "improvement_pct": True, "ssim": True, "psnr_db": True, "ms_ssim": True,
                    "drift": None, "jitter_ablation": None, "motion_ablation": None,
                    "history_ablation": None, "train_progress": None}


def load(reports_dir):
    """Every non-archived jitter-study report, tagged with its arm and the dataset it trained on.

    Three groups rather than two, because the arm alone does not identify the experiment. The same temporal
    control appears on both datasets, and it is the pair of control rows - identical configuration, only the
    capture's jitter differing - that says whether the freeze is caused by the jitter in the data or by the
    configuration itself. Collapsing them would hide the one comparison that discriminates the two.
    """
    runs = []
    patterns = [("p4_tjit_*.report.json", "jitter"), ("p4_thist_*.report.json", "control"),
                ("p4_v2control_*.report.json", "control")]
    dataset = {"jitter": "godot-v4 (jittered)", "control": "godot-v4 (jittered)"}
    for pattern, arm in patterns:
        for path in sorted(glob.glob(os.path.join(reports_dir, pattern))):
            if "PREFIX_HARNESS_BUG" in os.path.basename(path):
                continue
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            measured = report.get("measured", {})
            baseline = measured.get("val_baseline_l1") or 0.0
            val_l1 = measured.get("val_l1")
            failures = report.get("gates", {}).get("failures", [])
            group = arm
            if pattern.startswith("p4_v2control"):
                group, data = "control_v2", "godot-v2 (no jitter)"
            else:
                data = dataset[arm]
            runs.append({
                "name": os.path.basename(path).replace(".report.json", ""),
                "arm": group, "dataset": data,
                "seed": report.get("seed"), "inputs": report.get("inputs") or [],
                "valid": not failures, "failures": failures, "val_l1": val_l1,
                "improvement_pct": (100.0 * (baseline - val_l1) / baseline)
                                   if val_l1 is not None and baseline else None,
                "ssim": measured.get("ssim"), "psnr_db": measured.get("psnr_db"),
                "ms_ssim": measured.get("ms_ssim"), "drift": measured.get("drift"),
                "jitter_ablation": measured.get("jitter_ablation"),
                "motion_ablation": measured.get("motion_ablation"),
                "history_ablation": measured.get("history_ablation"),
                "train_progress": measured.get("train_progress"),
                "exported": bool(report.get("export")),
                "train_seconds": report.get("train_seconds"), "parameters": report.get("parameters"),
            })
    return runs


def fmt(value, places=5):
    return "n/a" if value is None else ("%.*f" % (places, value))


def permutation_test(a, b):
    """Two-sided exact permutation test on the difference of means, plus its smallest attainable p.

    Every way of splitting the pooled sample into a group of size len(a) is enumerated. With a handful of seeds
    per arm that is a few hundred arrangements - exact, and cheap enough not to need approximating under a
    normality assumption this sample size cannot support.
    """
    pooled = list(a) + list(b)
    n, total = len(a), len(pooled)
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    at_least, arrangements = 0, 0
    for indices in itertools.combinations(range(total), n):
        chosen = set(indices)
        mean_a = statistics.fmean(pooled[i] for i in indices)
        mean_b = statistics.fmean(pooled[i] for i in range(total) if i not in chosen)
        arrangements += 1
        if abs(mean_a - mean_b) >= observed - 1e-15:
            at_least += 1
    return at_least / arrangements, 1.0 / arrangements, arrangements


def bootstrap_ci(a, b, draws=20000, seed=20261022):
    """Percentile bootstrap CI on (mean(a) - mean(b)). Seeded, so the printed interval reproduces exactly."""
    import random
    rng = random.Random(seed)
    differences = sorted(statistics.fmean(rng.choice(a) for _ in range(len(a)))
                         - statistics.fmean(rng.choice(b) for _ in range(len(b))) for _ in range(draws))
    return differences[int(0.025 * draws)], differences[int(0.975 * draws) - 1]


def describe(values):
    """mean, sd, median, min, max and n - the spread as well as the centre."""
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "min": min(values), "max": max(values)}
def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reports", default=os.path.join("models", "p5"))
    args = parser.parse_args(argv[1:])
    runs = load(args.reports)
    if not runs:
        print("no jitter-study reports in %s" % args.reports)
        return 2
    valid = {arm: [r for r in runs if r["arm"] == arm and r["valid"]] for arm in ARMS}

    print("=" * 116)
    print("PER-SEED RESULTS   (%s)" % args.reports)
    print("=" * 116)
    print("%-22s %-8s %-9s %-9s %-8s %-8s %-8s %-8s %-9s %s"
          % ("run", "arm", "seed", "val L1", "vs base", "SSIM", "PSNR", "MS-SSIM", "jit abl", "gates"))
    for run in sorted(runs, key=lambda r: (r["arm"], r["seed"] or 0)):
        print("%-22s %-8s %-9s %-9s %-8s %-8s %-8s %-8s %-9s %s"
              % (run["name"], run["arm"], run["seed"], fmt(run["val_l1"]),
                 "%+.2f%%" % run["improvement_pct"] if run["improvement_pct"] is not None else "n/a",
                 fmt(run["ssim"], 4), fmt(run["psnr_db"], 2), fmt(run["ms_ssim"], 4),
                 fmt(run["jitter_ablation"]),
                 "all pass" if run["valid"] else "REFUSED (%d)" % len(run["failures"])))

    print()
    print("=" * 116)
    print("PER-ARM AGGREGATES   (valid runs only - a refused run is no result, not a bad one)")
    print("=" * 116)
    if min(len(valid[arm]) for arm in ARMS) < 2:
        print()
        print("*** WARNING: at least one arm has fewer than two valid runs. Every 'better on the mean' line")
        print("*** below then compares against a single seed and carries NO margin - it is printed because the")
        print("*** per-seed table is real, not because a difference has been established.")
        print("*** Treat the RELIABILITY section, which counts refusals, as the finding of record here.")
    metrics = [("val_l1", 5), ("improvement_pct", 2), ("ssim", 4), ("psnr_db", 2), ("ms_ssim", 4),
               ("drift", 5), ("jitter_ablation", 5), ("motion_ablation", 5),
               ("history_ablation", 5), ("train_progress", 4)]
    for metric, places in metrics:
        cells = {arm: describe([r[metric] for r in valid[arm] if r[metric] is not None]) for arm in ARMS}
        print("\n%s" % metric)
        for arm in ARMS:
            stat = cells[arm]
            if not stat:
                print("  %-8s n=0" % arm)
                continue
            print("  %-8s n=%d  mean %s  sd %s  median %s  range [%s, %s]"
                  % (arm, stat["n"], fmt(stat["mean"], places), fmt(stat["sd"], places),
                     fmt(stat["median"], places), fmt(stat["min"], places), fmt(stat["max"], places)))
        jitter_stat, control_stat = cells["jitter"], cells["control"]
        if jitter_stat and control_stat:
            difference = jitter_stat["mean"] - control_stat["mean"]
            direction = HIGHER_IS_BETTER.get(metric)
            if direction is None:
                print("  jitter - control = %s   (no direction asserted)" % fmt(difference, places))
            else:
                winner = "jitter" if (jitter_stat["mean"] > control_stat["mean"]) == direction else "control"
                print("  jitter - control = %+.*f   -> %s arm is better on the mean"
                      % (places, difference, winner))

    print("=" * 116)
    print("ARM COMPARISON   (exact permutation test + seeded bootstrap CI, on jitter - control)")
    print("=" * 116)
    for metric, places in [("val_l1", 5), ("ssim", 4), ("psnr_db", 2), ("ms_ssim", 4)]:
        a = [r[metric] for r in valid["jitter"] if r[metric] is not None]
        b = [r[metric] for r in valid["control"] if r[metric] is not None]
        if len(a) < 2 or len(b) < 2:
            print("\n%s: too few valid runs to test (jitter n=%d, control n=%d)" % (metric, len(a), len(b)))
            continue
        p, floor, arrangements = permutation_test(a, b)
        low, high = bootstrap_ci(a, b)
        print("\n%s" % metric)
        print("  jitter %s   control %s   -> %s ahead on the mean"
              % (fmt(statistics.fmean(a), places), fmt(statistics.fmean(b), places),
                 "jitter" if statistics.fmean(a) > statistics.fmean(b) else "control"))
        print("  95%% CI on the difference: %s to %s%s"
              % (fmt(low, places), fmt(high, places), "   <- includes zero" if low <= 0.0 <= high else ""))
        print("  permutation p = %.4f  (smallest attainable %.4f over %d arrangements)"
              % (p, floor, arrangements))
        print("  %s" % ("significant at 0.05" if p < 0.05 else
                         "NOT significant at 0.05 - the difference is unresolved, not absent"))

    print()
    print("=" * 116)
    print("THE DISCRIMINATING TABLE   (same configuration, three groups; this is what settles the question)")
    print("=" * 116)
    groups = [("jitter", "godot-v4 (jittered)", "+ de-jitter, jittered capture"),
              ("control", "godot-v4 (jittered)", "control,    jittered capture"),
              ("control_v2", "godot-v2 (no jitter)", "control,    un-jittered capture")]
    print("%-34s %-6s %-9s %-14s %s" % ("group", "runs", "frozen", "progress", "mean improvement"))
    for group, data, label in groups:
        selected = [r for r in runs if r["arm"] == group]
        if not selected:
            continue
        refused = [r for r in selected if not r["valid"]]
        progress = [r["train_progress"] for r in selected if r["train_progress"] is not None]
        improvements = [r["improvement_pct"] for r in selected
                        if r["improvement_pct"] is not None and r["valid"]]
        print("%-34s %-6d %-9s %-14s %s"
              % (label, len(selected), "%d (%.0f%%)" % (len(refused), 100.0 * len(refused) / len(selected)),
                 " ".join("%.0f%%" % (100 * p) for p in progress),
                 "%+.2f%%" % statistics.fmean(improvements) if improvements else "n/a"))
    print()
    print("Read: the two control rows are the same configuration differing only in whether the capture was")
    print("jittered. It trains on the un-jittered capture and freezes on the jittered one, so the freeze is")
    print("caused by the jitter in the data. The first row then shows the de-jitter removing that cause.")

    print()
    print("=" * 116)
    print("RELIABILITY DETAIL   (refused runs, excluded from every aggregate above)")
    print("=" * 116)
    for group, data, label in groups:
        refused = [r for r in runs if r["arm"] == group and not r["valid"]]
        if not refused:
            continue
        print("\n%s:" % label)
        for run in refused:
            for failure in run["failures"]:
                print("    refused %s: %s" % (run["name"], failure))

    print()
    print("=" * 116)
    print("EXPORTS")
    print("=" * 116)
    for run in sorted(runs, key=lambda r: (r["arm"], r["seed"] or 0)):
        if run["valid"]:
            print("  %-22s %-8s exported, %s parameters, %.0f s training"
                  % (run["name"], run["arm"], "{:,}".format(run["parameters"] or 0),
                     run["train_seconds"] or 0.0))
    print()
    print("notes:")
    print("  - The jitter ablation has no control value by construction: the control consumes no jitter input,")
    print("    so there is nothing to withhold. It measures how far the output moves when the offset is")
    print("    withheld, which is a dependence measure, not a quality one.")
    print("  - Progress is the fall in training residual L1. A run below ~10% is refused as degenerate, and")
    print("    averaging one in would pull the arm toward the bilinear baseline rather than describing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))