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

    # Final validation pass with the checker on the saved files.
    for name in ("nrr_upscaler_v0.1.onnx", "nrr_passthrough_2x.onnx"):
        m = onnx.load(os.path.join(models, name))
        onnx.checker.check_model(m)
        print(f"checker OK: {name}")


if __name__ == "__main__":
    main()
