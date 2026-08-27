import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "convlstm_phase_gated_simvp.py"
BASELINE_MODEL_PATH = ROOT / "models" / "convlstm_simvp.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_config(**overrides):
    config = {
        "in_channels": 1,
        "window": 3,
        "horizon": 3,
        "height": 8,
        "width": 16,
        "selected_channels": [0],
        "convlstm_hidden_dim": 16,
        "spatial_hidden_dim": 32,
        "temporal_hidden_dim": 64,
        "num_temporal_blocks": 3,
        "dropout": 0.1,
        "phase_context_dim": 32,
        "initial_history_weight": 0.7,
        "initial_translation_weight": 0.7,
    }
    config.update(overrides)
    return config


class ConvLSTMPhaseGatedSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (
            load_module("convlstm_phase_gated_simvp", MODEL_PATH)
            if MODEL_PATH.exists()
            else None
        )
        cls.baseline_module = load_module(
            "convlstm_simvp_baseline", BASELINE_MODEL_PATH
        )

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("phase-gated model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(
            MODEL_PATH.is_file(),
            "missing model file: models/convlstm_phase_gated_simvp.py",
        )

    def test_matches_platform_dry_run_shape(self):
        model = self.module.build_model(model_config()).eval()
        x = torch.randn(2, 3, 1, 8, 16)
        ls = torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]])

        with torch.no_grad():
            output = model(x, ls)

        self.assertEqual(tuple(output.shape), (2, 3, 1, 8, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_supports_multichannel_different_horizon_and_odd_spatial_size(self):
        config = model_config(
            in_channels=2,
            window=4,
            horizon=2,
            convlstm_hidden_dim=4,
            spatial_hidden_dim=6,
            temporal_hidden_dim=8,
            num_temporal_blocks=1,
            dropout=0.0,
            phase_context_dim=8,
        )
        model = self.module.build_model(config).eval()

        with torch.no_grad():
            output = model(
                torch.randn(2, 4, 2, 9, 15),
                torch.tensor(
                    [[0.0, 45.0, 90.0, 135.0], [180.0, 225.0, 270.0, 315.0]]
                ),
            )

        self.assertEqual(tuple(output.shape), (2, 2, 1, 9, 15))
        self.assertTrue(torch.isfinite(output).all())

    def test_supports_single_frame_window(self):
        model = self.module.build_model(
            model_config(
                window=1,
                horizon=2,
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                phase_context_dim=8,
            )
        ).eval()

        with torch.no_grad():
            output = model(torch.randn(2, 1, 1, 7, 9), torch.tensor([[0.0], [90.0]]))

        self.assertEqual(tuple(output.shape), (2, 2, 1, 7, 9))

    def test_accepts_floating_ls_dtypes_without_linear_dtype_errors(self):
        model = self.module.build_model(
            model_config(
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                phase_context_dim=8,
            )
        ).eval()

        try:
            with torch.no_grad():
                output = model(
                    torch.randn(2, 3, 1, 8, 16),
                    torch.tensor(
                        [[0.0, 90.0, 180.0], [45.0, 135.0, 225.0]],
                        dtype=torch.float64,
                    ),
                )
        except RuntimeError as error:
            self.fail(f"floating ls dtype reached Linear unchanged: {error}")

        self.assertEqual(tuple(output.shape), (2, 3, 1, 8, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_phase_context_encoder_builds_first_and_second_harmonics(self):
        encoder = self.module.PhaseContextEncoder(window=4, context_dim=8)
        ls = torch.tensor([[0.0, 90.0, 180.0, 270.0]])

        features = encoder.build_harmonic_features(ls)

        expected = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, -1.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0, -1.0],
                ]
            ]
        )
        self.assertEqual(tuple(features.shape), (1, 4, 4))
        self.assertTrue(torch.allclose(features, expected, atol=1e-6))

    def test_memory_gate_returns_bounded_gate_with_expected_shape(self):
        gate = self.module.PhaseConditionedMemoryGate(6, 8, 0.7)
        encoded = torch.randn(2, 4, 6, 5, 8)
        context = torch.randn(2, 8)

        admitted, gate_map = gate(encoded, context)

        self.assertEqual(tuple(admitted.shape), tuple(encoded.shape))
        self.assertEqual(tuple(gate_map.shape), (2, 4, 6, 5, 8))
        self.assertTrue(torch.isfinite(gate_map).all())
        self.assertTrue(((gate_map >= 0.0) & (gate_map <= 1.0)).all())

    def test_residual_gate_returns_bounded_gate_with_expected_shape(self):
        gate = self.module.PhaseConditionedResidualGate(6, 8, 0.7)
        future = torch.randn(2, 3, 6, 5, 8)
        last_state = torch.randn(2, 6, 5, 8)
        context = torch.randn(2, 8)

        final_state, gate_map = gate(future, last_state, context)

        self.assertEqual(tuple(final_state.shape), tuple(future.shape))
        self.assertEqual(tuple(gate_map.shape), (2, 3, 6, 5, 8))
        self.assertTrue(torch.isfinite(gate_map).all())
        self.assertTrue(((gate_map >= 0.0) & (gate_map <= 1.0)).all())

    def test_gate_initialization_matches_configured_probabilities(self):
        initial_history_weight = 0.65
        initial_translation_weight = 0.8
        memory_gate = self.module.PhaseConditionedMemoryGate(
            6, 8, initial_history_weight
        )
        residual_gate = self.module.PhaseConditionedResidualGate(
            6, 8, initial_translation_weight
        )

        for gate, expected_weight in (
            (memory_gate, initial_history_weight),
            (residual_gate, initial_translation_weight),
        ):
            expected_bias = torch.full_like(
                gate.gate.bias, torch.logit(torch.tensor(expected_weight))
            )
            self.assertTrue(torch.allclose(gate.gate.bias, expected_bias))
            self.assertEqual(torch.count_nonzero(gate.phase_to_bias.bias).item(), 0)

            for weight in (gate.gate.weight, gate.phase_to_bias.weight):
                fan_in = weight.shape[1]
                fan_out = weight.shape[0]
                bound = 0.1 * (6.0 / (fan_in + fan_out)) ** 0.5
                self.assertLessEqual(
                    weight.detach().abs().max().item(), bound + 1e-7
                )
                self.assertGreater(torch.count_nonzero(weight).item(), 0)

    def test_different_ls_changes_context_and_both_gates_for_same_features(self):
        torch.manual_seed(17)
        model = self.module.build_model(
            model_config(
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                phase_context_dim=8,
            )
        ).eval()
        x = torch.randn(2, 3, 1, 8, 16)
        ls_a = torch.zeros(2, 3)
        ls_b = torch.full((2, 3), 90.0)

        with torch.no_grad():
            recurrent = model.convlstm(x)
            encoded = model.spatial_encoder(
                recurrent.reshape(2 * 3, 4, 8, 16)
            ).reshape(2, 3, 6, 4, 8)
            context_a = model.phase_context_encoder(ls_a)
            context_b = model.phase_context_encoder(ls_b)
            admitted_a, memory_a = model.memory_gate(encoded, context_a)
            admitted_b, memory_b = model.memory_gate(encoded, context_b)
            future = model.temporal_translator(
                admitted_a.reshape(2, 3 * 6, 4, 8)
            ).reshape(2, 3, 6, 4, 8)
            _, residual_a = model.residual_gate(future, encoded[:, -1], context_a)
            _, residual_b = model.residual_gate(future, encoded[:, -1], context_b)

        self.assertFalse(torch.equal(context_a, context_b))
        self.assertFalse(torch.equal(memory_a, memory_b))
        self.assertFalse(torch.equal(residual_a, residual_b))
        self.assertFalse(torch.equal(admitted_a, admitted_b))

    def test_backward_reaches_phase_encoder_and_both_gates(self):
        torch.manual_seed(23)
        model = self.module.build_model(
            model_config(
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                phase_context_dim=8,
            )
        )
        output = model(
            torch.randn(2, 3, 1, 9, 15),
            torch.tensor([[0.0, 45.0, 90.0], [120.0, 180.0, 240.0]]),
        )

        output.square().mean().backward()

        for prefix in ("phase_context_encoder", "memory_gate", "residual_gate"):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, f"no parameters found for {prefix}")
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(
                any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
                f"all gradients are zero for {prefix}",
            )

    def test_all_trainable_parameters_get_finite_gradients(self):
        torch.manual_seed(31)
        model = self.module.build_model(
            model_config(
                in_channels=2,
                window=4,
                horizon=2,
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                phase_context_dim=8,
            )
        )
        output = model(
            torch.randn(2, 4, 2, 9, 15),
            torch.tensor(
                [[0.0, 30.0, 60.0, 90.0], [120.0, 180.0, 240.0, 300.0]]
            ),
        )
        output.mean().backward()

        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing gradient: {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                f"non-finite gradient: {name}",
            )

    def test_rejects_invalid_x_and_ls_inputs_with_named_errors(self):
        model = self.module.build_model(model_config())
        valid_x = torch.randn(2, 3, 1, 8, 16)
        valid_ls = torch.zeros(2, 3)

        invalid_calls = [
            ("x", lambda: model(torch.randn(2, 3, 8, 16), valid_ls)),
            ("window", lambda: model(torch.randn(2, 2, 1, 8, 16), valid_ls)),
            ("channels", lambda: model(torch.randn(2, 3, 2, 8, 16), valid_ls)),
            ("ls", lambda: model(valid_x)),
            ("ls", lambda: model(valid_x, torch.zeros(2, 3, 1))),
            ("ls", lambda: model(valid_x, torch.zeros(2, 2))),
            ("ls", lambda: model(valid_x, torch.zeros(2, 3, dtype=torch.long))),
            ("ls", lambda: model(valid_x, torch.tensor([[0.0, 1.0, float("nan")]]).expand(2, -1))),
            ("ls", lambda: model(valid_x, torch.tensor([[0.0, 1.0, float("inf")]]).expand(2, -1))),
            ("device", lambda: model(valid_x, torch.empty(2, 3, device="meta"))),
        ]

        for field, call in invalid_calls:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    call()

    def test_exports_complete_platform_schema(self):
        expected_parameters = {
            "convlstm_hidden_dim": {"type": "int", "default": 16, "min": 4, "max": 128},
            "spatial_hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
            "temporal_hidden_dim": {"type": "int", "default": 64, "min": 8, "max": 512},
            "num_temporal_blocks": {"type": "int", "default": 3, "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
            "phase_context_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
            "initial_history_weight": {"type": "float", "default": 0.7, "min": 0.01, "max": 0.99},
            "initial_translation_weight": {"type": "float", "default": 0.7, "min": 0.01, "max": 0.99},
        }
        expected_auxiliary_inputs = {
            "ls": {
                "required": True,
                "shape": ["batch", "window"],
                "dtype": "float32",
                "unit": "degree",
            }
        }

        self.assertEqual(self.module.MODEL_SPEC["parameters"], expected_parameters)
        self.assertEqual(
            self.module.MODEL_SPEC["auxiliary_inputs"], expected_auxiliary_inputs
        )
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_preserves_c_s_1_public_main_path_structure_and_dimensions(self):
        config = model_config(dropout=0.0)
        model = self.module.build_model(config)
        baseline = self.baseline_module.build_model(config)

        for name in (
            "convlstm",
            "spatial_encoder",
            "temporal_translator",
            "spatial_decoder",
        ):
            candidate_module = getattr(model, name)
            baseline_module = getattr(baseline, name)
            candidate_signature = [
                (
                    type(layer).__name__,
                    tuple(layer.weight.shape),
                    getattr(layer, "kernel_size", None),
                    getattr(layer, "stride", None),
                    getattr(layer, "padding", None),
                    getattr(layer, "groups", None),
                )
                for layer in candidate_module.modules()
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d))
            ]
            baseline_signature = [
                (
                    type(layer).__name__,
                    tuple(layer.weight.shape),
                    getattr(layer, "kernel_size", None),
                    getattr(layer, "stride", None),
                    getattr(layer, "padding", None),
                    getattr(layer, "groups", None),
                )
                for layer in baseline_module.modules()
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d))
            ]
            self.assertEqual(candidate_signature, baseline_signature, name)

        baseline_parameters = self.baseline_module.MODEL_SPEC["parameters"]
        for name, metadata in baseline_parameters.items():
            self.assertEqual(self.module.MODEL_SPEC["parameters"][name], metadata)

        candidate_tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
        baseline_tree = ast.parse(BASELINE_MODEL_PATH.read_text(encoding="utf-8"))
        main_path_classes = {
            "ConvLSTMCell",
            "ConvLSTMEncoder",
            "SpatialEncoder",
            "TemporalInceptionBlock",
            "TemporalTranslator",
            "SpatialDecoder",
        }
        candidate_classes = {
            node.name: ast.dump(node, include_attributes=False)
            for node in candidate_tree.body
            if isinstance(node, ast.ClassDef) and node.name in main_path_classes
        }
        baseline_classes = {
            node.name: ast.dump(node, include_attributes=False)
            for node in baseline_tree.body
            if isinstance(node, ast.ClassDef) and node.name in main_path_classes
        }
        self.assertEqual(candidate_classes, baseline_classes)

    def test_uses_only_torch_imports_and_upload_safe_calls(self):
        tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
        import_roots = set()
        called_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                import_roots.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)

        banned_calls = {
            "open",
            "eval",
            "exec",
            "compile",
            "__import__",
            "system",
            "popen",
            "Popen",
            "run",
        }
        self.assertEqual(import_roots, {"torch"})
        self.assertFalse(called_names & banned_calls)


if __name__ == "__main__":
    unittest.main()
