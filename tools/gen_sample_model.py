#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# gen_sample_model.py
# Generates the NRR reference sample models (ONNX) used by Phase 3 tests.
#
#  models/nrr_upscaler_v0.1.onnx    3-input upscaler matching
#                                   models/architecture.md section 3.1:
#     color  [1,3,H,W] float32 (dynamic H/W)
#     depth  [1,1,H,W] float32
#     motion [1,2,H,W] float32
#     output [1,3,2H,2W] float32 (2x upscale)
#
#  models/nrr_passthrough_2x.onnx   1-input generic fixture used by tests:
#     x      [1,3,H,W] float32
#     y      [1,3,2H,2W] float32
#
#  models/nrr_history_probe.onnx    reports its own `history` input as its output, so
#                                   tests/unit/test_history_delivery.cpp can read back the tensor the
#                                   runtime fed a model.
#
#  models/nrr_scale_token_probe.onnx         reports its own `scale` input - the resolution token,
#                                            log2(input_width / 128) - as its output, scaled by 0.25 so the
#  models/nrr_unrecognised_input_probe.onnx  value survives the 8-bit readback. The second is the same
#                                            graph with that input named something TensorRole does not
#                                            know, so tests/unit/test_scale_token.cpp can pin the refusal
#                                            as well as the delivery.
#
# Behavior: output == bilinear 2x upsample of the color input. The neural
# refinement path exists in the graph but its final convolution is
# zero-initialized, so the models are provably identity-preserving upscaling
# fixtures (gray stays gray, gradients stay smooth) without trained weights.
#
# Requires: pip install onnx numpy
# ---------------------------------------------------------------------------

import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _conv_weight(rng, out_c, in_c, k, scale=0.05):
    return (rng.standard_normal((out_c, in_c, k, k)) * scale).astype(np.float32)


def _make_upscaler(path, model_name, model_version, description,
                   with_depth_motion=True):
    rng = np.random.RandomState(20260910)

    if with_depth_motion:
        color = helper.make_tensor_value_info(
            "color", TensorProto.FLOAT, [1, 3, "H", "W"])
        depth = helper.make_tensor_value_info(
            "depth", TensorProto.FLOAT, [1, 1, "H", "W"])
        motion = helper.make_tensor_value_info(
            "motion", TensorProto.FLOAT, [1, 2, "H", "W"])
        inputs = [color, depth, motion]
    else:
        color = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, [1, 3, "H", "W"])
        inputs = [color]
    output = helper.make_tensor_value_info(
        "output" if with_depth_motion else "y",
        TensorProto.FLOAT, [1, 3, "H2", "W2"])

    initializers = []
    nodes = []
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "upscale_scales")

    if with_depth_motion:
        # --- Feature extraction (color) -------------------------------------
        w1 = _conv_weight(rng, 16, 3, 3)
        b1 = np.zeros(16, dtype=np.float32)
        initializers += [
            numpy_helper.from_array(w1, "conv_color.weight"),
            numpy_helper.from_array(b1, "conv_color.bias"),
        ]
        nodes += [
            helper.make_node("Conv", ["color", "conv_color.weight",
                                       "conv_color.bias"],
                             ["feat_color"], name="conv_color",
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["feat_color"], ["feat_color_act"],
                             name="relu_color"),
        ]

        # --- Depth fusion ---------------------------------------------------
        w2 = _conv_weight(rng, 8, 1, 3)
        b2 = np.zeros(8, dtype=np.float32)
        initializers += [
            numpy_helper.from_array(w2, "conv_depth.weight"),
            numpy_helper.from_array(b2, "conv_depth.bias"),
        ]
        nodes.append(helper.make_node(
            "Conv", ["depth", "conv_depth.weight", "conv_depth.bias"],
            ["feat_depth"], name="conv_depth",
            kernel_shape=[3, 3], pads=[1, 1, 1, 1]))

        # --- Motion fusion --------------------------------------------------
        w3 = _conv_weight(rng, 8, 2, 3)
        b3 = np.zeros(8, dtype=np.float32)
        initializers += [
            numpy_helper.from_array(w3, "conv_motion.weight"),
            numpy_helper.from_array(b3, "conv_motion.bias"),
        ]
        nodes.append(helper.make_node(
            "Conv", ["motion", "conv_motion.weight", "conv_motion.bias"],
            ["feat_motion"], name="conv_motion",
            kernel_shape=[3, 3], pads=[1, 1, 1, 1]))

        # --- Concat: 16 + 8 + 8 = 32 channels -------------------------------
        nodes.append(helper.make_node(
            "Concat", ["feat_color_act", "feat_depth", "feat_motion"],
            ["features"], name="concat_features", axis=1))
        features_src = "features"
    else:
        # Single-input fixture: identity 1x1 conv (preserves color).
        w0 = np.zeros((3, 3, 1, 1), dtype=np.float32)
        for c in range(3):
            w0[c, c, 0, 0] = 1.0
        initializers.append(numpy_helper.from_array(w0, "conv_identity.weight"))
        nodes.append(helper.make_node(
            "Conv", ["x", "conv_identity.weight"], ["feat"],
            name="conv_identity", kernel_shape=[1, 1]))
        features_src = "feat"

    # --- Upsample features and color 2x -------------------------------------
    initializers.append(scales)
    nodes.append(helper.make_node(
        "Resize", [features_src, "", "upscale_scales"], ["feat_up"],
        name="resize_features", mode="linear",
        coordinate_transformation_mode="half_pixel"))
    up_src = "x" if not with_depth_motion else "color"
    nodes.append(helper.make_node(
        "Resize", [up_src, "", "upscale_scales"], ["color_up"],
        name="resize_color", mode="linear",
        coordinate_transformation_mode="half_pixel"))

    # --- Refinement (zero-initialized residual path) ------------------------
    feat_channels = 32 if with_depth_motion else 3
    w4 = np.zeros((feat_channels, feat_channels, 3, 3), dtype=np.float32)
    b4 = np.zeros(feat_channels, dtype=np.float32)
    initializers += [
        numpy_helper.from_array(w4, "refine1.weight"),
        numpy_helper.from_array(b4, "refine1.bias"),
    ]
    nodes += [
        helper.make_node("Conv", ["feat_up", "refine1.weight", "refine1.bias"],
                         ["refine1"], name="refine1",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["refine1"], ["refine1_act"],
                         name="relu_refine1"),
    ]

    w5 = np.zeros((3, feat_channels, 3, 3), dtype=np.float32)
    b5 = np.zeros(3, dtype=np.float32)
    initializers += [
        numpy_helper.from_array(w5, "refine2.weight"),
        numpy_helper.from_array(b5, "refine2.bias"),
    ]
    nodes.append(helper.make_node(
        "Conv", ["refine1_act", "refine2.weight", "refine2.bias"],
        ["refine_rgb"], name="refine2",
        kernel_shape=[3, 3], pads=[1, 1, 1, 1]))

    # --- Output projection --------------------------------------------------
    out_name = "output" if with_depth_motion else "y"
    nodes.append(helper.make_node("Add", ["color_up", "refine_rgb"],
                                  [out_name], name="output_projection"))

    graph = helper.make_graph(
        nodes,
        "nrr_upscaler_v0_1" if with_depth_motion else "nrr_passthrough_2x",
        inputs, [output], initializer=initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model.producer_name = "NRR gen_sample_model.py"
    model.model_version = 1
    model.doc_string = description
    meta = model.metadata_props.add()
    meta.key = "model_name"
    meta.value = model_name
    meta = model.metadata_props.add()
    meta.key = "model_version"
    meta.value = model_version
    meta = model.metadata_props.add()
    meta.key = "description"
    meta.value = description
    meta = model.metadata_props.add()
    meta.key = "tags"
    meta.value = "upscaling,temporal" if with_depth_motion else "upscaling"

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}")
    print(f"  inputs:  {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in inputs]}")
    print(f"  outputs: {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in [output]]}")


def _make_history_probe(path, model_name, model_version, description):
    """A fixture whose output is the `history` input it was given, 2x nearest-neighbour replicated.

    Written for tests/unit/test_history_delivery.cpp, which needs to observe the tensor the runtime hands a
    model rather than the rule it hands it by. The output is the history plane itself, so the frame the
    runtime displays carries the delivered plane back to the caller - at exactly twice the size, and by
    replication rather than interpolation, so every output pixel (2i, 2j) is history[i, j] and decimating by
    two recovers the tensor exactly. That is the whole design: an interpolating 2x (the other fixtures)
    would make the readback lossy, and a lossy readback cannot distinguish a warp from a rounding.

    `history` is declared for the runtime, not used by the graph's colour path: a model with one input has
    its only input classified as colour (classify_tensor_role is overridden for single-input models), so a
    history-consuming fixture has to declare a second input to have the history one recognised at all.
    """
    color = helper.make_tensor_value_info("color", TensorProto.FLOAT, [1, 3, "H", "W"])
    history = helper.make_tensor_value_info("history", TensorProto.FLOAT, [1, 3, "H", "W"])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "H2", "W2"])

    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
                                     "probe_scales")
    # "asymmetric" + "floor" is integer replication: out[i] = in[floor(i / 2)], with no half-pixel shift to
    # make the decimation recover a different pixel than the one that was fed.
    node = helper.make_node("Resize", ["history", "", "probe_scales"], ["y"],
                            name="history_passthrough", mode="nearest",
                            coordinate_transformation_mode="asymmetric",
                            nearest_mode="floor")

    graph = helper.make_graph([node], "nrr_history_probe", [color, history], [output],
                              initializer=[scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model.producer_name = "NRR gen_sample_model.py"
    model.model_version = 1
    model.doc_string = description
    for key, value in (("model_name", model_name), ("model_version", model_version),
                       ("description", description), ("tags", "upscaling,temporal")):
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}")
    print(f"  inputs:  {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in (color, history)]}")
    print(f"  outputs: {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in (output,)]}")


def _make_scale_token_probe(path, model_name, model_version, description, token_input_name="scale"):
    """A fixture that reports the value of its own resolution-token input, 2x nearest-neighbour replicated.

    Written for tests/unit/test_scale_token.cpp. The released scale-agnostic model *consumes* the token
    (models/phase4/upscale_msreal_scale.onnx, inputs `color,jitter,scale`), so nothing about a render of it says
    what value arrived; this fixture publishes it instead, and the 0.25 multiplier is the whole trick - the
    token is log2(width / 128), so an unscaled 1.0 would saturate the 8-bit readback and a token of 2.0 would
    be indistinguishable from it. Quartered, 0.0/1.0/2.0 land at bytes 0/64/128 and a tier between them
    (192 -> 0.585 -> byte 37) is readable too.

    The graph is `scale * 0.25`, replicated to three channels and doubled - the same 2x integer replication the
    history probe uses, so decimating the displayed frame recovers the value exactly rather than through an
    interpolation.

    `color` is declared for the runtime and not consumed by the graph: a single-input model has its only input
    classified as colour, so a fixture whose token input is to be recognised by name has to declare a second
    input for that name to be reached at all (the same reason models/nrr_history_probe.onnx declares one).
    Passing a `token_input_name` the classifier does not recognise produces the control fixture, which exists so
    the refusal can be asserted rather than described.
    """
    color = helper.make_tensor_value_info("color", TensorProto.FLOAT, [1, 3, "H", "W"])
    # One channel, static - the declaration models/phase4/upscale_msreal_scale.onnx ships, which is what makes a
    # three-channel fill of this input a refusal (of the element count here, of the dimension in ONNX Runtime)
    # rather than a silent substitution.
    token = helper.make_tensor_value_info(token_input_name, TensorProto.FLOAT, [1, 1, "H", "W"])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "H2", "W2"])

    quarter = numpy_helper.from_array(np.array([0.25], dtype=np.float32), "probe_quarter")
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "probe_scales")
    nodes = [
        helper.make_node("Mul", [token_input_name, "probe_quarter"], ["scaled"], name="token_times_quarter"),
        # The runtime publishes a three-channel frame, so the value has to reach all three channels to be
        # readable; a one-channel output would exercise a conversion no caller sees.
        helper.make_node("Concat", ["scaled", "scaled", "scaled"], ["grey"], name="token_to_three_channels",
                         axis=1),
        # "asymmetric" + "floor" is integer replication: out[i] = in[floor(i / 2)], with no half-pixel shift.
        helper.make_node("Resize", ["grey", "", "probe_scales"], ["y"], name="token_passthrough",
                         mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
    ]

    graph = helper.make_graph(nodes, "nrr_scale_token_probe", [color, token], [output],
                              initializer=[quarter, scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model.producer_name = "NRR gen_sample_model.py"
    model.model_version = 1
    model.doc_string = description
    for key, value in (("model_name", model_name), ("model_version", model_version),
                       ("description", description), ("tags", "upscaling,scale-agnostic")):
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}")
    print(f"  inputs:  {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in (color, token)]}")
    print(f"  outputs: {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in (output,)]}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    models = os.path.normpath(os.path.join(here, os.pardir, "models"))
    os.makedirs(models, exist_ok=True)

    _make_upscaler(
        os.path.join(models, "nrr_upscaler_v0.1.onnx"),
        model_name="nrr_upscaler_v0.1",
        model_version="0.1.0",
        description="NRR reference neural upscaler (untrained skeleton, "
                    "identity-preserving 2x bilinear upsample).",
        with_depth_motion=True)

    _make_upscaler(
        os.path.join(models, "nrr_passthrough_2x.onnx"),
        model_name="nrr_passthrough_2x",
        model_version="0.1.0",
        description="NRR single-input generic upscaler fixture "
                    "(identity-preserving 2x bilinear upsample).",
        with_depth_motion=False)

    _make_history_probe(
        os.path.join(models, "nrr_history_probe.onnx"),
        model_name="nrr_history_probe",
        model_version="0.1.0",
        description="NRR history-delivery fixture: output is the `history` input, 2x nearest-neighbour "
                    "replicated, so a caller can read back exactly the tensor the runtime fed the model.")

    _make_scale_token_probe(
        os.path.join(models, "nrr_scale_token_probe.onnx"),
        model_name="nrr_scale_token_probe",
        model_version="0.1.0",
        description="NRR resolution-token fixture: output is the `scale` input (log2(input_width / 128)) times "
                    "0.25, 2x nearest-neighbour replicated, so a caller can read back exactly the value the "
                    "runtime fed the model's token input.")

    _make_scale_token_probe(
        os.path.join(models, "nrr_unrecognised_input_probe.onnx"),
        model_name="nrr_unrecognised_input_probe",
        model_version="0.1.0",
        description="Control for the resolution-token fixture: the same graph with the token input named "
                    "`zzz_unknown`, which TensorRole does not know, so the runtime's refusal can be asserted "
                    "instead of described.",
        token_input_name="zzz_unknown")

    # Final validation pass with the checker on the saved files.
    for name in ("nrr_upscaler_v0.1.onnx", "nrr_passthrough_2x.onnx", "nrr_history_probe.onnx",
                 "nrr_scale_token_probe.onnx", "nrr_unrecognised_input_probe.onnx"):
        m = onnx.load(os.path.join(models, name))
        onnx.checker.check_model(m)
        print(f"checker OK: {name}")


if __name__ == "__main__":
    main()
