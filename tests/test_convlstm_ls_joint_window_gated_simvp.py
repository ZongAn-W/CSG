import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "convlstm_ls_joint_window_gated_simvp.py"
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
        "horizon": 20,
        "height": 8,
        "width": 16,
        "selected_channels": [0],
        "convlstm_hidden_dim": 16,
        "spatial_hidden_dim": 32,
        "temporal_hidden_dim": 64,
        "num_temporal_blocks": 3,
        "dropout": 0.1,
        "window_gate_hidden_dim": 32,
        "initial_window_strength": 0.05,
    }
    config.update(overrides)
    return config


class ConvLSTMLsJointWindowGatedSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (
            load_module("convlstm_ls_joint_window_gated_simvp", MODEL_PATH)
            if MODEL_PATH.exists()
            else None
        )
        cls.baseline_module = load_module(
            "convlstm_simvp_joint_window_baseline", BASELINE_MODEL_PATH
        )

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("joint-window-gated model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(
            MODEL_PATH.is_file(),
            "missing model file: models/convlstm_ls_joint_window_gated_simvp.py",
        )

    def test_horizon_twenty_output_matches_platform_contract(self):
        model = self.module.build_model(model_config()).eval()
        x = torch.randn(2, 3, 1, 8, 16)
        ls = torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]])

        with torch.no_grad():
            output = model(x, ls)

        self.assertEqual(tuple(output.shape), (2, 20, 1, 8, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_supports_multichannel_different_horizon_and_odd_spatial_size(self):
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
                window_gate_hidden_dim=8,
            )
        ).eval()

        with torch.no_grad():
            output = model(
                torch.randn(2, 4, 2, 9, 15),
                torch.tensor(
                    [[0.0, 45.0, 90.0, 135.0], [180.0, 225.0, 270.0, 315.0]]
                ),
            )

        self.assertEqual(tuple(output.shape), (2, 2, 1, 9, 15))
        self.assertTrue(torch.isfinite(output).all())

    def test_supports_single_input_window(self):
        model = self.module.build_model(
            model_config(
                window=1,
                horizon=2,
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                window_gate_hidden_dim=8,
            )
        ).eval()

        with torch.no_grad():
            output = model(torch.randn(2, 1, 1, 7, 9), torch.tensor([[0.0], [90.0]]))

        self.assertEqual(tuple(output.shape), (2, 2, 1, 7, 9))

    def test_builds_correct_per_window_ls_harmonics(self):
        gate = self.module.LsConditionedJointWindowGate(4, 6, 8, 0.05)
        features = gate.build_harmonic_features(
            torch.tensor([[0.0, 90.0, 180.0, 270.0]])
        )
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

    def test_returns_independent_bounded_weights_and_near_identity_scales(self):
        torch.manual_seed(11)
        gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 0.05)
        encoded = torch.randn(2, 3, 6, 5, 8)
        ls = torch.tensor([[0.0, 45.0, 90.0], [135.0, 180.0, 225.0]])

        weighted, weights, scales = gate(encoded, ls)

        self.assertEqual(tuple(weighted.shape), tuple(encoded.shape))
        self.assertEqual(tuple(weights.shape), (2, 3))
        self.assertEqual(tuple(scales.shape), (2, 3))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(((weights > 0.0) & (weights < 1.0)).all())
        self.assertFalse(torch.allclose(weights.sum(dim=1), torch.ones(2)))
        self.assertGreater(scales.detach().min().item(), 0.95)
        self.assertLess(scales.detach().max().item(), 1.05)

    def test_each_weight_depends_on_other_windows_and_their_ls(self):
        torch.manual_seed(13)
        gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 0.05).eval()
        encoded = torch.randn(1, 3, 6, 5, 8)
        ls = torch.tensor([[0.0, 45.0, 90.0]])

        with torch.no_grad():
            original = gate.window_weights(encoded, ls)
            for source_step in range(3):
                changed_encoded = encoded.clone()
                changed_encoded[:, source_step] += 1.0
                feature_changed = gate.window_weights(changed_encoded, ls)
                self.assertTrue(((original - feature_changed).abs() > 0.0).all())

                changed_ls = ls.clone()
                changed_ls[:, source_step] += 60.0
                ls_changed = gate.window_weights(encoded, changed_ls)
                self.assertTrue(((original - ls_changed).abs() > 0.0).all())

    def test_final_input_window_is_actively_scaled(self):
        gate = self.module.LsConditionedJointWindowGate(3, 2, 4, 0.2).eval()
        with torch.no_grad():
            gate.scorer[-1].weight.zero_()
            gate.scorer[-1].bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
        encoded = torch.ones(1, 3, 2, 2, 2)

        with torch.no_grad():
            weighted, _, scales = gate(encoded, torch.zeros(1, 3))

        self.assertFalse(torch.allclose(scales[:, -1], torch.ones(1)))
        self.assertFalse(torch.equal(weighted[:, -1], encoded[:, -1]))

    def test_initialization_matches_strength_and_small_output_gain(self):
        gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 0.05)

        self.assertAlmostEqual(gate.current_strength().item(), 0.05, places=6)
        self.assertEqual(torch.count_nonzero(gate.scorer[-1].bias).item(), 0)
        weight = gate.scorer[-1].weight.detach()
        bound = 0.01 * (6.0 / (weight.shape[0] + weight.shape[1])) ** 0.5
        self.assertLessEqual(weight.abs().max().item(), bound + 1e-7)
        self.assertGreater(torch.count_nonzero(weight).item(), 0)

    def test_strength_endpoints_are_exact_and_zero_is_identity(self):
        encoded = torch.randn(2, 3, 6, 5, 8)
        ls = torch.tensor([[0.0, 45.0, 90.0], [135.0, 180.0, 225.0]])

        zero_gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 0.0)
        one_gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 1.0)
        zero_weighted, _, zero_scales = zero_gate(encoded, ls)

        self.assertEqual(zero_gate.current_strength().item(), 0.0)
        self.assertEqual(one_gate.current_strength().item(), 1.0)
        self.assertTrue(torch.equal(zero_scales, torch.ones_like(zero_scales)))
        self.assertTrue(torch.equal(zero_weighted, encoded))

    def test_strength_can_recover_after_optimizer_steps_outside_bounds(self):
        for initial_raw, target in ((-0.01, 0.5), (1.01, 0.5)):
            with self.subTest(initial_raw=initial_raw):
                gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 0.05)
                with torch.no_grad():
                    gate.window_strength.fill_(initial_raw)
                optimizer = torch.optim.SGD([gate.window_strength], lr=0.1)

                optimizer.zero_grad()
                loss = (gate.current_strength() - target).square()
                loss.backward()
                gradient = gate.window_strength.grad.detach().item()
                optimizer.step()

                self.assertNotEqual(gradient, 0.0)
                self.assertGreaterEqual(gate.current_strength().item(), 0.0)
                self.assertLessEqual(gate.current_strength().item(), 1.0)
                self.assertGreater(gate.window_strength.item(), 0.0)
                self.assertLess(gate.window_strength.item(), 1.0)

    def test_same_features_with_different_ls_change_window_weights(self):
        torch.manual_seed(17)
        gate = self.module.LsConditionedJointWindowGate(3, 6, 8, 0.05).eval()
        encoded = torch.randn(2, 3, 6, 5, 8)

        with torch.no_grad():
            weights_a = gate.window_weights(encoded, torch.zeros(2, 3))
            weights_b = gate.window_weights(encoded, torch.full((2, 3), 90.0))

        self.assertFalse(torch.equal(weights_a, weights_b))

    def test_backward_reaches_every_joint_gate_parameter(self):
        torch.manual_seed(23)
        model = self.module.build_model(
            model_config(
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                window_gate_hidden_dim=8,
            )
        )
        output = model(
            torch.randn(2, 3, 1, 9, 15),
            torch.tensor([[0.0, 45.0, 90.0], [120.0, 180.0, 240.0]]),
        )
        output.square().mean().backward()

        for name, parameter in model.window_gate.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing gradient: {name}")
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(
                torch.count_nonzero(parameter.grad).item(), 0, f"zero gradient: {name}"
            )

    def test_all_trainable_parameters_get_finite_gradients(self):
        torch.manual_seed(29)
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
                window_gate_hidden_dim=8,
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
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_accepts_float64_ls_and_rejects_invalid_inputs(self):
        model = self.module.build_model(model_config())
        valid_x = torch.randn(2, 3, 1, 8, 16)
        valid_ls = torch.zeros(2, 3)

        with torch.no_grad():
            output = model(valid_x, valid_ls.to(torch.float64))
        self.assertEqual(tuple(output.shape), (2, 20, 1, 8, 16))

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

    def test_rejects_invalid_configuration(self):
        invalid_configs = [
            ("positive integer", model_config(window_gate_hidden_dim=0)),
            ("initial_window_strength", model_config(initial_window_strength=-0.1)),
            ("initial_window_strength", model_config(initial_window_strength=1.1)),
            ("dropout", model_config(dropout=1.0)),
        ]
        for message, config in invalid_configs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.module.build_model(config)

    def test_exports_complete_platform_schema(self):
        expected_parameters = {
            "convlstm_hidden_dim": {"type": "int", "default": 16, "min": 4, "max": 128},
            "spatial_hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
            "temporal_hidden_dim": {"type": "int", "default": 64, "min": 8, "max": 512},
            "num_temporal_blocks": {"type": "int", "default": 3, "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
            "window_gate_hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
            "initial_window_strength": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0},
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

    def test_c_s_1_main_path_classes_are_ast_identical(self):
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
        baseline_parameters = self.baseline_module.MODEL_SPEC["parameters"]
        for name, metadata in baseline_parameters.items():
            self.assertEqual(self.module.MODEL_SPEC["parameters"][name], metadata)

    def test_uses_only_torch_without_attention_softmax_or_unsafe_calls(self):
        source = MODEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
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
            "open", "eval", "exec", "compile", "__import__",
            "system", "popen", "Popen", "run",
        }
        self.assertEqual(import_roots, {"torch"})
        self.assertFalse(called_names & banned_calls)
        self.assertNotIn("softmax", source.lower())
        self.assertNotIn("attention", source.lower())
        self.assertNotIn("einsum", source.lower())


if __name__ == "__main__":
    unittest.main()
