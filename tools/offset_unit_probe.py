#!/usr/bin/env python3
"""offset_unit_probe.py - the two units an offset travels through, and what composing them wrong costs.

`PhaseAlignedAccumulator::add_frame` scales its `offset` argument by `out_width / width`, so its convention is
*frame-grid* pixels: at the output resolution (the default source) that factor is one and the two units coincide,
which is why the distinction never had to be written down. `phase_aligned_frame_for` fills
`PhaseAlignedFrame::offset_x` in *output* pixels instead (pinned by
tests/unit/test_jitter.cpp, test_phase_aligned_offset_rule_matches_the_two_model_kinds: jitter 0.25 at a 2x
pipeline is 0.5). `TemporalAccumulator::apply` passed that value straight through, and the input-render source -
`NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER`, the one arrangement where the frame is coarser than the grid, and so the
only one where the two units can disagree - therefore placed each sample at `jitter * scale^2` in grid pixels
rather than at `jitter * scale` where the renderer took it.

This probe is the number behind that: the same splat of the same frames, once at each placement, scored against
the footprint-averaged zone plate. `tools/regen_aa_fixture.scatter_resolve` multiplies its offset by `scale`
internally, so `offset * scale` is exactly what the composed runtime did - the two rows below differ only in
that factor.

Usage:
    python tools/offset_unit_probe.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen_aa_fixture import (HI, LO, PLATE_K, edge_error, edge_weights, halton_offset,  # noqa: E402
                              point_frame, scatter_resolve, upsample_mirror, zone_plate_reference)

FRAMES = 8


def main():
    scale = HI // LO
    offsets = [halton_offset(index + 1) for index in range(FRAMES)]
    frames = [point_frame(offset, PLATE_K, LO) for offset in offsets]
    reference = zone_plate_reference(PLATE_K)
    weights = edge_weights(reference)

    bilinear = upsample_mirror(frames[0], HI)
    one_frame = scatter_resolve([frames[0]], [offsets[0]], 1.0, HI)
    taken_at = scatter_resolve(frames, offsets, 1.0, HI)
    composed = scatter_resolve(frames, [offset * scale for offset in offsets], 1.0, HI)

    rows = (("one frame, bilinear upsample", bilinear),
            ("one frame, splatted where taken", one_frame),
            ("%d frames, splatted at j*scale (what the capture asks for)" % FRAMES, taken_at),
            ("%d frames, splatted at j*scale^2 (what apply() passed)" % FRAMES, composed))

    print("offset unit: %d frames at scale %d, against the footprint-averaged zone plate" % (FRAMES, scale))
    print()
    for label, plane in rows:
        print("  %-58s edge %.6f  plain %.6f"
              % (label, edge_error(plane, reference, weights), float(np.abs(plane - reference).mean())))

    def against(a, b, key):
        edge = (edge_error(a, reference, weights) - edge_error(b, reference, weights))
        plain = (float(np.abs(a - reference).mean()) - float(np.abs(b - reference).mean()))
        return edge, plain

    print()
    edge, plain = against(taken_at, bilinear, "edge")
    print("  splatting where the samples were taken, against bilinear:  edge %+.1f%%, plain %+.1f%%"
          % (edge / edge_error(bilinear, reference, weights) * 100.0,
             plain / float(np.abs(bilinear - reference).mean()) * 100.0))
    edge, plain = against(composed, bilinear, "edge")
    print("  splatting at twice the displacement, against bilinear:     edge %+.1f%%, plain %+.1f%%"
          % (edge / edge_error(bilinear, reference, weights) * 100.0,
             plain / float(np.abs(bilinear - reference).mean()) * 100.0))
    edge, plain = against(composed, taken_at, "edge")
    print("  so the composed unit cost:                                 edge %+.1f%%, plain %+.1f%%"
          % (edge / edge_error(taken_at, reference, weights) * 100.0,
             plain / float(np.abs(taken_at - reference).mean()) * 100.0))
    print()
    distance = float(np.abs(scatter_resolve([frames[0]], [offsets[0]], 1.0, HI)
                            - scatter_resolve([frames[0]], [offsets[0] * scale], 1.0, HI)).max())
    print("  the two placements of the first frame alone differ by %.6f (max): visible on a signal that" % distance)
    print("  varies, and invisible on the constant input render the arrangement test uses, which is why the")
    print("  test that caught the copy-back grid could not see this one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
