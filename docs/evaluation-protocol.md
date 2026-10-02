# Evaluation protocol

Pre-registered **before** the P2 frontier decision (ch32 vs ch64) and **before** any P3 quality-lever
result. Every bar and rule below was fixed in advance, so a number that arrives later cannot be used to
rationalise the decision it now judges. The commit that introduces this file is the timestamp of that
fixing.

## What is measured, and what each metric is trusted for

| dimension | metric | trusted for |
| --- | --- | --- |
| reconstruction | L1 (mean absolute error) | the primary gate - the trainer's loss and the number the gates already assert |
| reconstruction | PSNR | comparability with published numbers **only** - see the note below |
| structural | SSIM, MS-SSIM | whether structure is preserved (mirrored against `runtime/nrr_quality.cpp` for SSIM) |
| perceptual | LPIPS, DISTS | whether the image *reads* right (pretrained backbones) |
| video | VMAF (+ motion feature) | whole-sequence quality, the commercial dimension |
| temporal | warping error, temporal PSNR/SSIM | whether the reconstruction shimmers |
| detail | high-frequency-energy ratio | whether the reconstruction keeps the detail the input lost |
| performance | per-tier latency, ms/frame | whether the model costs a frame budget |

**PSNR is reported, not gated.** Mean-squared error rewards the smooth, low-variance bilinear output: on a
held-out scene the model beat bilinear on every structural/perceptual/detail metric while PSNR was *slightly
lower* (27.30 vs 27.72 dB). A bar that the sharper, better model fails is a bad bar, so PSNR appears in the
report for published-number comparability and does not decide anything.

## Scenes

* **Validation (the quality claim):** `heldout`, `heldout2` from `models/training-data/godot-v2`.
* **Robustness / overfit check:** `train`, `train2`, `train3`. These are seen during training, so they are
  *not* a quality claim. They exist to catch memorisation: if validation quality is good but train quality
  is far better, the model memorised rather than generalised, and that is a finding, not a pass.

## Seeds

* **Frontier probe:** 2 seeds (`20261020`, `20261021`), matching the ch32 runs already on disk.
* **Final model:** 10 seeds. The noise floor put σ ≈ 2.4 points on held-out L1 improvement, and two seeds of
  ch32 already span 7.3 points - so two seeds cannot separate configurations. A final claim needs ten.

## Resolution tiers

* **Native (scored):** 128 → 256, the trained size and the only tier with reference pairs. All quality
  metrics are scored here.
* **Dynamic smoke check:** 160 → 320. The export already verifies the graph is dynamic in H/W; this is a
  *can it run* check, not a quality claim, because there is no reference data at 160.
* **Runtime latency tiers:** 256→512, 540p→1080p, 1080p→4K, via `tools/measure_model.py`. Latency only -
  there is no reference data at these sizes to score quality against.

## Pass/fail bars (pre-registered, every bar must hold on every validation scene)

| dimension | bar |
| --- | --- |
| reconstruction | L1 ≤ baseline L1 × 0.95 (≥ 5% better - the existing gate) |
| structural | SSIM > baseline SSIM **and** MS-SSIM > baseline MS-SSIM |
| perceptual | LPIPS < baseline LPIPS **and** DISTS < baseline DISTS |
| video | VMAF > baseline VMAF |
| detail | detail ratio > baseline detail ratio |
| temporal | warping error ≤ reference warping error × 1.25 (no wild shimmer), and temporal SSIM ≥ reference × 0.95 |
| performance | within the per-tier budgets in `docs/roadmap.md`, and model latency ≤ ch32 latency × 1.5 |

A bar fails the *model*, but a bar whose own measurement is unavailable fails *the harness*: an unavailable
metric is reported as `available: false` with a reason and the run is marked incomplete, never silently
passed.

## Frontier rule (ch32 vs ch64)

`ch64` (3.7M parameters) is adopted over `ch32` (94k) only if, across the two seeds, **both** hold:

1. ch64's mean held-out L1 improvement exceeds ch32's by **≥ 5 points** - the same magnitude as the gate's
   `MIN_IMPROVEMENT`, so the extra capacity has to buy at least another 5% over what 32 channels already
   deliver.
2. ch64 passes every bar above on every validation scene, and is not worse than ch32 on SSIM, MS-SSIM,
   LPIPS, DISTS, VMAF, detail ratio, or warping error.

The 5-point threshold is not arbitrary: σ ≈ 2.4 points on L1 improvement and ch32's own two-seed span of
7.3 points mean a smaller margin is indistinguishable from noise. Paying 39× parameters for a gain inside
the noise is the same trap the colour-only decision already refused. If ch64 does not clear both, `ch32`
stands and P3 proceeds on it.

## P3 quality levers, gated by the same protocol

Each lever is adopted only if it improves held-out L1 by ≥ 5 points over the current config across ≥ 2
seeds **and** passes every bar above. Levers, in the order they are tried (each carries the chosen config
forward if adopted, otherwise the config reverts):

1. **Input set** (colour vs colour+depth+motion+history) - re-probed at the chosen capacity, because the
   earlier colour-only decision was measured at ch32 and the point of more capacity is that it may finally
   have room to use motion/history.
2. **Loss function** (L1 → Charbonnier, and L1 → L1 + SSIM term).
3. **Learning-rate schedule** (cosine decay, and warmup length).
4. **Depth/detail weighting** (upweight high-frequency regions via the detail-ratio field).

No lever changes the protocol: the same scenes, seeds, tiers and bars judge every lever, and a lever that
wins on L1 alone but regresses a structural/perceptual metric is refused, because a single-number win is
exactly what this protocol exists to prevent.