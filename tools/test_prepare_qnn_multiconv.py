# SPDX-License-Identifier: MIT
"""Focused host contracts for staging the nine-Conv experiment."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import prepare_qnn_multiconv as experiment
from prepare_qnn_probe import qdq_conv


def source_and_replacements():
    nodes, inputs, outputs, initializers, replacements = [], [], [], [], {}
    for index, spec in enumerate(experiment.TARGETS):
        value = np.full(spec["weight_shape"], .001 * (index + 1), np.float32)
        bias = np.linspace(-.1, .1, 64, dtype=np.float32)
        inputs.append(helper.make_tensor_value_info(spec["input"], TensorProto.FLOAT, spec["input_shape"]))
        outputs.append(helper.make_tensor_value_info(spec["output"], TensorProto.FLOAT, spec["output_shape"]))
        initializers.extend([numpy_helper.from_array(value, spec["weight"]), numpy_helper.from_array(bias, spec["bias"])])
        nodes.append(helper.make_node("Conv", [spec["input"], spec["weight"], spec["bias"]], [spec["output"]],
                                      name=spec["node"], **spec["attributes"]))
        replacements[spec["id"]] = qdq_conv(tuple(spec["input_shape"]), value, bias, spec["attributes"], (-2, 2), (-4, 4))[0]
    # State passthroughs and a native normalization outside every selected cut
    # must survive verbatim, independently of the new quantized nodes.
    for i in range(21):
        inputs.append(helper.make_tensor_value_info(f"state_{i:03d}", TensorProto.FLOAT, [1]))
        outputs.append(helper.make_tensor_value_info(f"next_state_{i:03d}", TensorProto.FLOAT, [1]))
        nodes.append(helper.make_node("Identity", [f"state_{i:03d}"], [f"next_state_{i:03d}"], name=f"state_identity{i}"))
    inputs.append(helper.make_tensor_value_info("native_norm_input", TensorProto.FLOAT, [1, 1, 1, 64]))
    outputs.append(helper.make_tensor_value_info("native_norm_output", TensorProto.FLOAT, [1, 64, 1, 1]))
    initializers.extend([numpy_helper.from_array(np.ones(64, np.float32), "native_gamma"),
                         numpy_helper.from_array(np.zeros(64, np.float32), "native_beta"),
                         numpy_helper.from_array(np.full((64, 1, 1), -.25, np.float32), "native_slope")])
    nodes.append(helper.make_node("ChannelNormPRelu", ["native_norm_input", "native_gamma", "native_beta", "native_slope"],
                                  ["native_norm_output"], name="preserved_native_norm", domain="com.reuse", epsilon=1e-5))
    graph = helper.make_graph(nodes, "independent_conv_cuts_and_states", inputs, outputs, initializer=initializers)
    source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)], ir_version=10)
    return source, replacements


class MulticonvPreparationTest(unittest.TestCase):
    def test_exact_nine_branch_stage_targets_and_causal_shapes(self):
        specs = experiment.target_specs()
        self.assertEqual(len(specs), 9)
        self.assertEqual(len({x["node"] for x in specs}), 9)
        self.assertEqual(len({x["id"] for x in specs}), 9)
        for branch in ("dense_encoder", "mask_decoder_list.2", "phase_decoder_list.2"):
            matches = [x for x in specs if x["node"].startswith("/" + branch + "/")]
            self.assertEqual(len(matches), 3)
            self.assertEqual([x["weight_shape"][1] for x in matches], [128, 192, 256])
            for block, spec in zip((1, 2, 3), matches):
                self.assertEqual(spec["node"], f"/{branch}/dense_block/dense_block.{block}.1/Conv")
                dy, dx = spec["attributes"]["dilations"]
                _, _, height, width = spec["input_shape"]
                # Independent valid 3x3 convolution dimensions, temporal padding0/frequency padding1.
                self.assertEqual(height - 2 * dy, 5)
                self.assertEqual(width + 2 - 2 * dx, spec["output_shape"][3])
        self.assertIn("reuse_encoder_dilated_conv_u8", {x["id"] for x in specs})

    def test_graft_preserves_all_other_nodes_initializers_and_public_state_abi(self):
        source, replacements = source_and_replacements()
        original = source.SerializeToString()
        candidate = experiment.graft_convolutions(source, replacements)
        self.assertEqual(source.SerializeToString(), original)
        self.assertEqual(list(candidate.graph.input), list(source.graph.input))
        self.assertEqual(list(candidate.graph.output), list(source.graph.output))
        names = {x["node"] for x in experiment.TARGETS}
        old_nodes = {x.name: x.SerializeToString() for x in source.graph.node if x.name not in names}
        new_nodes = {x.name: x.SerializeToString() for x in candidate.graph.node}
        self.assertTrue(all(new_nodes[name] == value for name, value in old_nodes.items()))
        self.assertTrue(names.isdisjoint(new_nodes))
        old_weights = {x.name: x.SerializeToString() for x in source.graph.initializer}
        new_weights = {x.name: x.SerializeToString() for x in candidate.graph.initializer}
        self.assertTrue(all(new_weights[name] == value for name, value in old_weights.items()))
        self.assertEqual(sum(x.op_type == "Conv" for x in candidate.graph.node), 9)
        self.assertEqual(len([x for x in candidate.graph.input if x.name.startswith("state_")]), 21)
        information = {x.name: x for x in candidate.graph.value_info}
        for spec in experiment.TARGETS:
            # These cuts are graph IO in this independent test model, so their
            # metadata remains in graph.input/output rather than value_info.
            prefix = "/qnn_multiconv/" + spec["id"] + "/"
            self.assertTrue(any(name.startswith(prefix) for name in information))
        onnx.checker.check_model(candidate, full_check=True)

    def test_scope_rejects_extra_nonconv_operations_and_incompatible_io(self):
        source, replacements = source_and_replacements()
        spec = experiment.TARGETS[0]
        modified = copy.deepcopy(replacements)
        modified[spec["id"]].graph.node.append(helper.make_node("Identity", ["input"], ["outside_scope"], name="injected_normalization_bypass"))
        with self.assertRaisesRegex(ValueError, "outside Conv/QDQ"):
            experiment.graft_convolutions(source, modified)
        modified = copy.deepcopy(replacements)
        modified[spec["id"]].graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
        with self.assertRaisesRegex(ValueError, "shape differs"):
            experiment.graft_convolutions(source, modified)

    def test_eight_conv_selection_keeps_excluded_target_byte_exact(self):
        source, replacements = source_and_replacements()
        excluded = "/dense_encoder/dense_block/dense_block.2.1/Conv"
        specs = experiment.select_targets([excluded])
        self.assertEqual(len(specs), 8)
        self.assertNotIn(excluded, {x["node"] for x in specs})
        selected_replacements = {spec["id"]: replacements[spec["id"]] for spec in specs}
        candidate = experiment.graft_convolutions(source, selected_replacements, specs)
        before = next(node for node in source.graph.node if node.name == excluded)
        after = next(node for node in candidate.graph.node if node.name == excluded)
        self.assertEqual(before.SerializeToString(), after.SerializeToString())
        self.assertEqual(list(source.graph.input), list(candidate.graph.input))
        self.assertEqual(list(source.graph.output), list(candidate.graph.output))
        self.assertEqual(len([node for node in candidate.graph.node if node.name.startswith("/qnn_multiconv/") and node.op_type == "Conv"]), 8)

    def test_exclusions_are_bounded_known_unique_and_nonempty(self):
        known = experiment.TARGETS[0]["node"]
        for exclusions in (["/unknown/Conv"], [known, known], [x["node"] for x in experiment.TARGETS]):
            with self.subTest(exclusions=exclusions):
                with self.assertRaises(ValueError): experiment.select_targets(exclusions)
        self.assertEqual(experiment.select_targets(), experiment.TARGETS)

    def test_clipping_reports_out_of_range_values_without_changing_quantization(self):
        value = np.array([-2, -1, 0, 1, 2], np.float32)
        quantization = {"scale": .01, "zero_point": 128}
        report = experiment.clipping_statistics(value, quantization)
        self.assertEqual(report["elements"], 5)
        self.assertEqual(report["below_range"], 1)
        self.assertEqual(report["above_range"], 1)
        self.assertEqual(report["clipped_fraction"], .4)
        self.assertEqual(report["maximum_excess_quantized_codes"], 73)
        self.assertEqual(quantization, {"scale": .01, "zero_point": 128})
        with self.assertRaises(ValueError): experiment.clipping_statistics(np.array([np.nan]), quantization)

    def test_selected_bundle_path_and_hash_are_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"full_stream": {"candidate": {"model": "../outside.onnx", "model_sha256": experiment.SINGLE_CONV_SHA256}}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsafe selected model path"):
                experiment.selected_model(root)
            manifest["full_stream"]["candidate"]["model"] = "bad.onnx"
            (root / "bad.onnx").write_bytes(b"not the selected model")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                experiment.selected_model(root)

    def test_wrong_source_hash_cannot_create_experiment_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "wrong.onnx"; graph.write_bytes(b"not F950")
            with self.assertRaisesRegex(ValueError, "source graph SHA-256 mismatch"):
                experiment.prepare(graph, root / "missing.wav", root / "missing.dll", root / "output", root / "selected")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__": unittest.main()
