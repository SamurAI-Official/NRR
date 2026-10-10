#!/usr/bin/env python3
"""export_runtime_plane_case.py - pin the runtime's input-render resolve against the Python derive.

The cross-check the derive cannot make for itself. `tools/refinement_base_dataset.py --plane phase-aligned-splat`
mirrors what `NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER` displays, and its placement is validated against two other
statements of the placement rule - but all three live in Python. The comparison that matters is against the
*runtime*: the same frames in, the same displayed bytes out. This tool builds a small, exactly-specified case
(a coarse render per frame, the engine's jitter, and a motion field whose left column leaves the frame so the
warp's emptying is exercised), computes the plane the *mirror* says the runtime will display, and writes both into
`tests/generated/plane_case.h`. The C++ test that consumes it
(`test_input_render_resolve_matches_the_derived_plane`, tests/integration/test_temporal_accumulation.cpp) uploads
those exact bytes through the real device path and compares every channel.

Why the case is small (4x8 input rendering to an 8x16 display grid): the generated expectation is stored as bytes
in a header, and this project's convention is that a pinned constant is checkable by reading it. Every value is
chosen to be exact in the formats it crosses - quarter- and eighth-pixel displacements, exact in both
float32/float64 and in the RG16F the runtime decodes - so a disagreement is a *rule* disagreement rather than a
rounding one. The tool refuses to write a case whose field does not survive the half-float round trip exactly.

    python tools/export_runtime_plane_case.py            # write the header
    python tools/export_runtime_plane_case.py --check    # fail if the checked-in header is stale

The unit the case's field is in is the frame contract's - pixel motion on the input grid, which is what the
runtime's motion texture carries - so the mirror's unit auto-detection is bypassed by declaring it, and the
generated header says which unit the case pins.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refinement_base_dataset as derive  # noqa: E402  (the mirror under test, not a second copy of it)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT_PATH = os.path.join(ROOT, "tests", "generated", "plane_case.h")

# The fixture's grids: the sample model is a 2x upscaler. Square, because the derive's splat mirror places onto a
# square grid (`scatter_sum`/`upsample_rgb` take one size per axis) and refuses anything else rather than
# mis-placing it - 8x8 renders to 16x16, the same *ratio* as the product's 128x128 to 256x256.
IN_W, IN_H = 8, 8
OUT_W, OUT_H = IN_W * 2, IN_H * 2

# Per frame: the engine's jitter in *frame-grid* pixels. Every value is a multiple of 1/8, so the placement
# fractions the splat computes are exact, and none is zero - a zero jitter is what makes a frame ineligible for
# the pass, and a case that integrates nothing pins nothing.
FRAMES = (1, 2, 3)
JITTER = ((0.125, -0.25), (-0.375, 0.125), (0.5, 0.25))

# How much of a frame may be decided by float32-vs-float64 rounding before the case is refused as
# non-discriminating (the measured case marks 10%, and only sixteen of those bytes actually differ), and how many
# tolerated differences the C++ test accepts before it stops believing the comparison is about the rule. The
# mirror is float64 and the runtime computes its taps, weights and resolve in float32, so a byte whose value sits
# within an ulp of the write-back's boundary can differ by one level with nothing wrong; every other byte must
# match exactly, which is where a placement, a warp or a fallback error shows up - as tens of levels, on most of
# the frame (a wrong offset unit moved values by 34 levels on 40% of the bytes, measured).
ROUNDING_TOLERANCE = 1e-4
AMBIGUOUS_LIMIT = 0.35


def color_bytes(frame_index):
    """The renderer's own coarse pass, as RGB bytes with alpha: steep, and different for every frame.

    A render with a strong gradient is where a placement error shows as a bias instead of hiding in flat content,
    and a per-frame offset means the accumulation carries content from more than one frame - which is what makes
    the warp visible at all.
    """
    out = np.zeros((IN_H, IN_W, 4), dtype=np.uint8)
    for row in range(IN_H):
        for col in range(IN_W):
            out[row, col, 0] = (17 * col + 29 * row + 37 * frame_index) % 256
            out[row, col, 1] = (61 * col + 7 * row + 11 * frame_index) % 256
            out[row, col, 2] = (5 * col + 53 * row + 97 * frame_index) % 256
            out[row, col, 3] = 255
    return out


def motion_field(frame_index):
    """Pixel motion on the input grid, quarter/eighth-pixel values, with the left column leaving the frame.

    The left column's horizontal component (0.5, 0.75, 1.0 input pixels per frame, so 1.0 to 2.0 output pixels at
    this 2x grid) puts the accumulated source outside the frame for the leftmost output pixels, which is the path
    that empties a pixel and hands it to the coverage fallback - the branch a case whose content only ever moved
    inside the frame would leave untested.
    """
    leave = 0.25 * (1 + frame_index)
    out = np.zeros((IN_H, IN_W, 2), dtype=np.float64)
    for row in range(IN_H):
        for col in range(IN_W):
            out[row, col, 0] = leave if col == 0 else 0.25 * (col - 1)
            out[row, col, 1] = 0.125 * (((row + frame_index) % 5) - 2)
    return out


def cases():
    """The case's inputs, and the planes the *mirror* says the runtime will display, frame by frame."""
    built = []
    state = None
    for position, frame_index in enumerate(FRAMES):
        colors = color_bytes(frame_index)
        field = motion_field(frame_index)
        # Every field value has to survive RG16F exactly, or the runtime would read a different number from the
        # one the mirror warped by, and the comparison would be about the encoder instead of the rule.
        for value in field.reshape(-1):
            if float(np.float16(value)) != float(value):
                raise SystemExit("the case's field value %.6f does not round-trip through half (frame %d): a "
                                 "case that crosses a format cannot pin a rule" % (value, frame_index))
        pair = {
            "input_clean": colors[..., :3].astype(np.float64) / 255.0,
            "jitter": np.array(JITTER[position], dtype=np.float64),
            "motion": field,
            # Only its shape is read - the output grid the resolve is on. The case carries no ground truth.
            "target": np.zeros((OUT_H, OUT_W, 3), dtype=np.float32),
        }
        plane, state, coverage = derive.phase_aligned_frame(pair, state, motion_unit="pixels")
        # The runtime's own write-back rule (interleaved_float_to_rgb8): clamp, scale by 255, add the half and
        # truncate. One definition of the byte the C++ test compares against.
        clipped = np.clip(plane, 0.0, 1.0)
        scaled = clipped * 255.0 + 0.5
        expected = scaled.astype(np.uint8)
        # A byte is *ambiguous* when its value sits on the write-back's rounding boundary: there the byte is
        # decided by precision - the runtime computes its taps, weights and resolve in float32 where this mirror
        # is float64 - and not by the rule, so the comparison has to tolerate one level there. This case's values
        # are exact dyadic combinations of the bytes (quarter- and eighth-pixel displacements), so about a fifth
        # of them land exactly on a boundary; everything else is pinned exactly, which is where a placement, a
        # warp or a fallback error would show. The measured case marks ~19% and only ~1.6% actually differ, and
        # the test additionally caps the tolerated count at 5% of the frame.
        ambiguous = np.where(np.abs(scaled - np.round(scaled)).reshape(-1) < ROUNDING_TOLERANCE)[0]
        if ambiguous.size > AMBIGUOUS_LIMIT * expected.size:
            raise SystemExit("frame %d: %d of %d expected bytes sit on the write-back's rounding boundary "
                             "(limit %.0f%%), so the case could not pin the rule"
                             % (frame_index, ambiguous.size, expected.size, AMBIGUOUS_LIMIT * 100.0))
        built.append({
            "index": frame_index,
            "jitter": JITTER[position],
            "input": colors.reshape(-1),
            "motion": field,
            "expected": expected.reshape(-1),
            "ambiguous": ambiguous,
            "coverage": coverage,
        })
    return built


def motion_bytes(case):
    """The RG16F bytes the motion texture holds: u then v, half floats, four bytes per pixel."""
    out = np.zeros((IN_H * IN_W, 4), dtype=np.uint8)
    for row in range(IN_H):
        for col in range(IN_W):
            pixel = row * IN_W + col
            for channel in (0, 1):
                half = np.uint16(np.float16(case["motion"][row, col, channel]).view(np.uint16))
                out[pixel, channel * 2] = half & 0xFF
                out[pixel, channel * 2 + 1] = (half >> 8) & 0xFF
    return out.reshape(-1)


def format_bytes(values, per_line=12, indent="    "):
    lines = []
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        lines.append(indent + "".join("0x%02x, " % int(value) for value in chunk))
    return "\n".join(lines)


def render_header(built):
    """The generated header: the case's bytes, and the plane the mirror says the runtime displays for each."""
    lines = [
        "/* Generated by tools/export_runtime_plane_case.py - do not edit by hand.",
        " *",
        " * The inputs are a coarse render per frame, the engine's jitter, and a motion field in the frame",
        " * contract's unit (pixel motion on the input grid). The expected planes are what",
        " * tools/refinement_base_dataset.py's input-render mirror says the runtime will display, so the test that",
        " * reads this header compares the runtime against the derive rather than against a second copy of a rule.",
        " *",
        " * Regenerate:  python tools/export_runtime_plane_case.py",
        " * Verify:      python tools/export_runtime_plane_case.py --check",
        " */",
        "#ifndef NRR_GENERATED_PLANE_CASE_H",
        "#define NRR_GENERATED_PLANE_CASE_H",
        "",
        "#include <cstdint>",
        "",
        "namespace nrr {",
        "namespace test {",
        "namespace plane_case {",
        "",
        "const uint32_t kInW = %d;" % IN_W,
        "const uint32_t kInH = %d;" % IN_H,
        "const uint32_t kOutW = %d;" % OUT_W,
        "const uint32_t kOutH = %d;" % OUT_H,
        "const uint32_t kFrameCount = %d;" % len(built),
        "/* The unit the case's field is in: the frame contract's, not a packer's. */",
        "const char kMotionUnit[] = \"pixel motion on the input grid\";",
        "",
        "struct Case {",
        "    uint32_t frame_index;",
        "    float jitter_x;",
        "    float jitter_y;",
        "    const uint8_t* input;    /* RGBA8, kInW x kInH: what the renderer submitted */",
        "    const uint8_t* motion;   /* RG16F, kInW x kInH: pixel motion on the input grid */",
        "    const uint8_t* expected; /* RGB8, kOutW x kOutH: what the runtime must display */",
        "    /* Bytes whose expected value sits on the write-back's rounding boundary (the mirror is float64, the",
        "     * runtime float32, so these may differ by one level). Everything else must match exactly. */",
        "    const uint16_t* ambiguous;",
        "    uint32_t ambiguous_count;",
        "};",
        "",
    ]
    for position, case in enumerate(built):
        lines.append("/* frame %d: jitter %.3f, %.3f; the first frame's samples reach %.3f of the grid; "
                     "%d ambiguous byte(s) */"
                     % (case["index"], case["jitter"][0], case["jitter"][1], case["coverage"],
                        case["ambiguous"].size))
        for name, values in (("kInput%d" % position, case["input"]),
                             ("kMotion%d" % position, motion_bytes(case)),
                             ("kExpected%d" % position, case["expected"])):
            lines.append("const uint8_t %s[] = {" % name)
            lines.append(format_bytes(values))
            lines.append("};")
        lines.append("const uint16_t kAmbiguous%d[] = {%s};"
                     % (position, "" if case["ambiguous"].size == 0 else
                        " " + ", ".join(str(int(value)) for value in case["ambiguous"]) + " "))
        lines.append("")
    lines.append("const Case kCases[] = {")
    for position, case in enumerate(built):
        lines.append("    {%d, %.3ff, %.3ff, kInput%d, kMotion%d, kExpected%d, kAmbiguous%d, %du},"
                     % (case["index"], case["jitter"][0], case["jitter"][1], position, position, position,
                        position, case["ambiguous"].size))
    lines += [
        "};",
        "",
        "} // namespace plane_case",
        "} // namespace test",
        "} // namespace nrr",
        "",
        "#endif /* NRR_GENERATED_PLANE_CASE_H */",
        "",
    ]
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Pin the runtime's input-render resolve against tools/refinement_base_dataset.py.")
    parser.add_argument("--out", default=OUT_PATH, help="the header to write (default: %s)" % OUT_PATH)
    parser.add_argument("--check", action="store_true",
                        help="write nothing, and fail if the file on disk differs from what would be generated")
    args = parser.parse_args(argv[1:])

    # The project's convention: the validations run first, and nothing is written unless they hold. This is the
    # derive's own placement check, against both statements of the rule already in the tree.
    loop_worst, probe_worst = derive.validate_placement()
    print("placement: the vectorised splat against the fixture's own loop %.3e, against the probe's %.3e"
          % (loop_worst, probe_worst))
    if max(loop_worst, probe_worst) > derive.PLACEMENT_TOLERANCE:
        raise SystemExit("the mirror's placement does not agree with the pinned one, so a case built from it "
                         "would pin the wrong rule")

    built = cases()
    text = render_header(built)
    total_ambiguous = 0
    for case in built:
        total_ambiguous += int(case["ambiguous"].size)
        print("frame %d: jitter %.3f, %.3f; first-frame coverage %.3f; expected plane %d bytes, %d ambiguous"
              % (case["index"], case["jitter"][0], case["jitter"][1], case["coverage"],
                 len(case["expected"]), case["ambiguous"].size))
    print("ambiguous bytes: %d of %d (%.2f%%), each allowed to differ by one level; the rest must match exactly"
          % (total_ambiguous, len(built) * len(built[0]["expected"]),
             100.0 * total_ambiguous / float(len(built) * len(built[0]["expected"]))))

    if args.check:
        if not os.path.exists(args.out):
            raise SystemExit("%s does not exist: run without --check to generate it" % args.out)
        with open(args.out, "r", encoding="utf-8") as handle:
            existing = handle.read()
        if existing != text:
            raise SystemExit("%s is stale: regenerate it with python tools/export_runtime_plane_case.py"
                             % args.out)
        print("%s matches what this tool generates" % args.out)
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
