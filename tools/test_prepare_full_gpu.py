# SPDX-License-Identifier: MIT
"""Focused guards for the experimental standard-FP32 asset preparer."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

import prepare_full_gpu as prepare


def identity_model(shape=(2, 3)):
    return helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")], "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def state_model_and_manifest():
    states = [{"input": f"state_{i:03d}", "output": f"next_state_{i:03d}",
               "shape": [1], "dtype": "float32", "label": f"state{i}"} for i in range(21)]
    pairs = [("magnitude", "spectrum_real", [5, 161]), ("phase", "spectrum_imag", [5, 161])]
    pairs += [(s["input"], s["output"], s["shape"]) for s in states]
    graph = helper.make_graph([helper.make_node("Identity", [a], [b], name=b) for a, b, _ in pairs],
                              "public_state_abi", [helper.make_tensor_value_info(a, TensorProto.FLOAT, s) for a, _, s in pairs],
                              [helper.make_tensor_value_info(b, TensorProto.FLOAT, s) for _, b, s in pairs])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    manifest = {**prepare.SETTINGS, "source_repository": "nvidia/Real-time_RE-USE",
                "source_revision": prepare.REVISION, "source_checkpoint_sha256": prepare.CHECKPOINT_SHA256,
                "commercial_use": False, "states": states}
    return model, manifest


class FullGpuPreparationTest(unittest.TestCase):
    def test_custom_domains_imports_functions_and_subgraphs_fail_closed(self):
        original = identity_model()
        candidates = []
        custom = copy.deepcopy(original); custom.graph.node[0].domain = "com.reuse"; candidates.append(custom)
        unused = copy.deepcopy(original); unused.opset_import.append(helper.make_opsetid("com.reuse", 1)); candidates.append(unused)
        nested = copy.deepcopy(original)
        nested.graph.node[0].attribute.append(helper.make_attribute("hidden_graph", identity_model().graph))
        candidates.append(nested)
        functions = copy.deepcopy(original); functions.functions.add(); candidates.append(functions)
        for model in candidates:
            with self.subTest(model=model.graph.node[0].domain, functions=len(model.functions)):
                with self.assertRaises(ValueError): prepare.validate_standard_graph(model)
        prepare.validate_standard_graph(original)

    def test_external_data_and_non_fp32_tensors_are_rejected_before_loading(self):
        for tensor_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE):
            model = identity_model()
            tensor = model.graph.initializer.add(name="bad", data_type=tensor_type, dims=[1])
            with self.subTest(dtype=tensor_type):
                with self.assertRaisesRegex(ValueError, "Non-FP32"):
                    prepare.validate_standard_graph(model)
        model = identity_model()
        tensor = model.graph.initializer.add(name="outside", data_type=TensorProto.FLOAT, dims=[1],
                                              data_location=TensorProto.EXTERNAL)
        tensor.external_data.add(key="location", value="../../must-not-load.bin")
        with self.assertRaisesRegex(ValueError, "External"):
            prepare.validate_standard_graph(model)
        model = identity_model()
        bad = helper.make_tensor("hidden", TensorProto.FLOAT, [1], [0.0])
        bad.data_location = TensorProto.EXTERNAL
        model.graph.node.append(helper.make_node("Constant", [], ["constant"], value=bad))
        with self.assertRaisesRegex(ValueError, "External"):
            prepare.validate_standard_graph(model)

    def test_quantized_ops_and_reduced_precision_cast_are_rejected(self):
        for op in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulInteger", "QLinearAdd"):
            model = identity_model(); model.graph.node[0].op_type = op
            with self.subTest(op=op):
                with self.assertRaisesRegex(ValueError, "Quantization"):
                    prepare.validate_standard_graph(model)
        model = identity_model()
        model.graph.node[0].CopyFrom(helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16))
        with self.assertRaisesRegex(ValueError, "casts"):
            prepare.validate_standard_graph(model)

    def test_dynamic_zero_or_non_float_public_io_is_rejected(self):
        for shape in (["batch", 3], [0, 3], []):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "static FP32"):
                    prepare.static_float_io(identity_model(shape))
        model = identity_model(); model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT64
        with self.assertRaisesRegex(ValueError, "static FP32"):
            prepare.static_float_io(model)

    def test_pinned_configuration_and_21_state_abi_are_enforced(self):
        model, manifest = state_model_and_manifest()
        prepare.validate_manifest(manifest, model, manifest)
        for key, value in (("sample_rate", 16000), ("exit_layers", 2), ("frames_per_call", 1),
                           ("look_ahead_frames", 1), ("source_checkpoint_sha256", "0"*64),
                           ("commercial_use", True)):
            wrong = copy.deepcopy(manifest); wrong[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): prepare.validate_manifest(wrong, model, manifest)
        wrong = copy.deepcopy(manifest); wrong["states"][0]["shape"] = [2]
        with self.assertRaisesRegex(ValueError, "state ABI"):
            prepare.validate_manifest(wrong, model, manifest)
        wrong_model = copy.deepcopy(model)
        del wrong_model.graph.output[-1]
        with self.assertRaisesRegex(ValueError, "state ABI"):
            prepare.validate_manifest(manifest, wrong_model, manifest)

    def test_initializer_guard_compares_values_shapes_and_multiplicity(self):
        model = identity_model()
        model.graph.initializer.append(numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "weight"))
        renamed = copy.deepcopy(model); renamed.graph.initializer[0].name = "exporter_renamed_weight"
        self.assertTrue(prepare.initializer_identity(model, renamed)["passed"])
        changed = copy.deepcopy(renamed)
        changed.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((2, 3), np.float32), "exporter_renamed_weight"))
        with self.assertRaisesRegex(ValueError, "initializer bytes"):
            prepare.initializer_identity(model, changed)
        reshaped = copy.deepcopy(renamed); reshaped.graph.initializer[0].dims[:] = [3, 2]
        with self.assertRaisesRegex(ValueError, "initializer bytes"):
            prepare.initializer_identity(model, reshaped)
        duplicate = copy.deepcopy(renamed); duplicate.graph.initializer.append(copy.deepcopy(renamed.graph.initializer[0]))
        with self.assertRaisesRegex(ValueError, "initializer bytes"):
            prepare.initializer_identity(model, duplicate)

    def test_closed_shape_folding_preserves_runtime_data_and_exact_output(self):
        model = identity_model()
        del model.graph.node[:]
        model.graph.node.extend([
            helper.make_node("Shape", ["x"], ["shape"], name="metadata_shape"),
            helper.make_node("Constant", [], ["flat_shape"], value=numpy_helper.from_array(np.array([6], np.int64))),
            helper.make_node("Reshape", ["x", "flat_shape"], ["flat"], name="runtime_data_reshape"),
            helper.make_node("Reshape", ["flat", "shape"], ["y"], name="runtime_data_restore")])
        original_bytes = model.SerializeToString()
        folded, report = prepare.fold_static_constants(model)
        self.assertEqual(model.SerializeToString(), original_bytes)
        self.assertEqual(report["shape_nodes_replaced"], 1)
        self.assertTrue(report["all_runtime_tensor_dimensions_static"])
        self.assertFalse(any(n.op_type in ("Shape", "Constant") for n in folded.graph.node))
        self.assertEqual(sum(n.op_type == "Reshape" for n in folded.graph.node), 2)
        data = np.arange(6, dtype=np.float32).reshape(2, 3) * -3.25
        runtimes = [ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]) for m in (model, folded)]
        np.testing.assert_array_equal(runtimes[0].run(None, {"x": data})[0], runtimes[1].run(None, {"x": data})[0])

    def test_runtime_slice_controls_cannot_be_mistaken_for_closed_constants(self):
        model = identity_model()
        # A scalar read from actual signal values controls Slice, even though
        # all public shapes are fixed. It must not be frozen by the preparer.
        del model.graph.node[:]
        model.graph.node.extend([
            helper.make_node("ReduceSum", ["x"], ["sum"], keepdims=1),
            helper.make_node("Cast", ["sum"], ["count2d"], to=TensorProto.INT64),
            helper.make_node("Constant", [], ["one"], value=numpy_helper.from_array(np.array([1], np.int64))),
            helper.make_node("Reshape", ["count2d", "one"], ["end"]),
            helper.make_node("Slice", ["x", "zero", "end", "axis"], ["y"])])
        model.graph.initializer.extend([numpy_helper.from_array(np.array([0], np.int64), "zero"),
                                         numpy_helper.from_array(np.array([0], np.int64), "axis")])
        with self.assertRaisesRegex(ValueError, "Slice controls|not static"):
            prepare.fold_static_constants(model)

    def test_closed_pad_controls_are_folded_before_downstream_shape_proof(self):
        graph = helper.make_graph([
            helper.make_node("Constant", [], ["four"], value=numpy_helper.from_array(np.array([4], np.int64))),
            helper.make_node("ConstantOfShape", ["four"], ["leading"],
                             value=numpy_helper.from_array(np.array([0], np.int64))),
            helper.make_node("Constant", [], ["trailing"],
                             value=numpy_helper.from_array(np.array([0, 0, 0, 2], np.int64))),
            helper.make_node("Concat", ["leading", "trailing"], ["pads"], axis=0),
            helper.make_node("Pad", ["x", "pads"], ["padded"]),
            helper.make_node("Conv", ["padded", "weights"], ["convolved"]),
            helper.make_node("PRelu", ["convolved", "slope"], ["activated"]),
            helper.make_node("Shape", ["activated"], ["output_shape"]),
            helper.make_node("Reshape", ["activated", "output_shape"], ["y"])], "pad_then_shape",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 5])],
            initializer=[numpy_helper.from_array(np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1), "weights"),
                         numpy_helper.from_array(np.full((2, 1, 1), .25, np.float32), "slope")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        folded, report = prepare.fold_static_constants(model)
        self.assertGreaterEqual(report["folding_iterations"], 2)
        self.assertFalse(any(n.op_type in ("ConstantOfShape", "Shape") for n in folded.graph.node))
        data = (np.arange(12, dtype=np.float32) - 5).reshape(1, 2, 2, 3)
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        runtimes = [ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
                    for m in (model, folded)]
        np.testing.assert_array_equal(runtimes[0].run(None, {"x": data})[0], runtimes[1].run(None, {"x": data})[0])

    def test_input_free_random_operation_is_not_a_closed_constant(self):
        model = identity_model()
        del model.graph.node[:]
        model.graph.node.extend([
            helper.make_node("RandomUniform", [], ["random"], shape=[2, 3], dtype=TensorProto.FLOAT),
            helper.make_node("Add", ["x", "random"], ["y"])])
        folded, _ = prepare.fold_static_constants(model)
        self.assertEqual(sum(n.op_type == "RandomUniform" for n in folded.graph.node), 1)
        self.assertNotIn("random", {x.name for x in folded.graph.initializer})

    def test_waveform_gate_requires_both_fixed_limits_and_full_state_accounting(self):
        valid = {"waveform": {"max_abs": .0009, "normalized_rmse": .0049},
                 "independent_state_feedback": True, "all_21_states_checked": True,
                 "all_values_finite": True, "samples": 51840, "calls": 66}
        self.assertTrue(prepare.waveform_gate(valid)["passed"])
        for key, value in (("max_abs", .0010001), ("normalized_rmse", .0050001),
                           ("max_abs", float("nan")), ("normalized_rmse", None)):
            wrong = copy.deepcopy(valid); wrong["waveform"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, "fixed waveform"):
                    prepare.waveform_gate(wrong)
        for key, value in (("independent_state_feedback", False), ("all_21_states_checked", False),
                           ("all_values_finite", False), ("samples", 51839), ("calls", 65)):
            wrong = copy.deepcopy(valid); wrong[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): prepare.waveform_gate(wrong)

    def test_output_cannot_overlap_inputs_or_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"; source.mkdir()
            fixtures = root / "fixtures"; fixtures.mkdir()
            occupied = root / "occupied"; occupied.mkdir()
            marker = occupied / "keep"; marker.write_bytes(b"preserve")
            for output in (source, source / "child", root, occupied):
                with self.subTest(output=output):
                    with self.assertRaises(ValueError): prepare.fresh_output(output, (source, fixtures))
            self.assertEqual(marker.read_bytes(), b"preserve")
            fresh = root / "fresh"
            self.assertEqual(prepare.fresh_output(fresh, (source, fixtures)), fresh.resolve())
            self.assertFalse(fresh.exists())

    def test_wrong_fixture_hash_fails_before_creating_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixtures = root / "fixtures"; fixtures.mkdir()
            (fixtures / "manifest.json").write_text("{}", encoding="utf-8")
            output = root / "new-assets"
            with self.assertRaisesRegex(ValueError, "fixture manifest.*SHA-256 mismatch"):
                prepare.prepare(root / "source", fixtures, output, root / "missing.dll")
            self.assertFalse(output.exists())
            self.assertEqual((fixtures / "manifest.json").read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
