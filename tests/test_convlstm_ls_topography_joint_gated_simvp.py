import ast
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "convlstm_ls_topography_joint_gated_simvp.py"
BASELINE_MODEL_PATH = ROOT / "models" / "convlstm_simvp.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_ast(module_path, class_name):
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"missing class {class_name} in {module_path}")


def trainable_parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def model_config(**overrides):
    config = {
        "in_channels": 5,
        "window": 20,
        "horizon": 20,
        "height": 8,
        "width": 16,
        "selected_channels": [0, 1, 2, 3, 4],
        "convlstm_hidden_dim": 16,
        "spatial_hidden_dim": 32,
        "temporal_hidden_dim": 64,
        "num_temporal_blocks": 3,
        "dropout": 0.1,
        "gate_hidden_dim": 32,
        "initial_gate_strength": 0.05,
    }
    config.update(overrides)
    return config


class ConvLSTMLsTopographyJointGatedSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (
            load_module("convlstm_ls_topography_joint_gated_simvp", MODEL_PATH)
            if MODEL_PATH.exists()
            else None
        )
        cls.baseline_module = load_module(
            "convlstm_simvp_ls_topography_baseline", BASELINE_MODEL_PATH
        )

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("LS-topography-gated model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(
            MODEL_PATH.is_file(),
            "missing model file: models/convlstm_ls_topography_joint_gated_simvp.py",
        )

    def test_exports_exact_auxiliary_metadata(self):
        self.assertEqual(
            self.module.MODEL_SPEC["auxiliary_inputs"],
            {
                "ls": {
                    "required": True,
                    "shape": ["batch", "window"],
                    "dtype": "float32",
                    "unit": "degree",
                },
                "topography": {
                    "required": True,
                    "shape": ["batch", 1, "height", "width"],
                    "dtype": "float32",
                    "unit": "meter",
                },
            },
        )

    def test_exports_baseline_parameters_plus_gate_parameters(self):
        expected = {
            "convlstm_hidden_dim",
            "spatial_hidden_dim",
            "temporal_hidden_dim",
            "num_temporal_blocks",
            "dropout",
            "gate_hidden_dim",
            "initial_gate_strength",
        }
        self.assertEqual(set(self.module.MODEL_SPEC["parameters"]), expected)

    def test_forward_requires_x_ls_then_topography(self):
        parameters = inspect.signature(
            self.module.ConvLSTMLsTopographyJointGatedSimVP.forward
        ).parameters
        self.assertEqual(list(parameters), ["self", "x", "ls", "topography"])

    def test_build_model_returns_module(self):
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_ls_harmonic_encoder_builds_expected_harmonic_features(self):
        encoder = self.module.LsHarmonicEncoder(hidden_dim=8)

        features = encoder.build_harmonic_features(
            torch.tensor([[0.0, 90.0, 180.0, 270.0]])
        )

        expected = torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, -1.0],
            ]
        )
        torch.testing.assert_close(features[0], expected, atol=1e-6, rtol=0.0)

    def test_ls_harmonic_encoder_forward_casts_float64_features_for_linear_layers(self):
        encoder = self.module.LsHarmonicEncoder(hidden_dim=8)
        ls = torch.tensor([[0.0, 90.0, 180.0]], dtype=torch.float64)

        encoded = encoder(ls)

        self.assertEqual(encoded.shape, (1, 3, 8))
        self.assertEqual(encoded.dtype, encoder.layers[0].weight.dtype)
        self.assertTrue(torch.isfinite(encoded).all())

    def test_topography_encoder_scales_and_resizes_finite_features(self):
        encoder = self.module.TopographyEncoder(hidden_dim=8)
        topography = torch.tensor([[[[-10_000.0, 0.0], [5_000.0, 10_000.0]]]])

        scaled = encoder.scale_elevation(topography)
        encoded = encoder(topography, output_size=(2, 2))

        torch.testing.assert_close(
            scaled, torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]])
        )
        self.assertEqual(encoded.shape, (1, 8, 2, 2))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_joint_gate_returns_encoded_scales_and_bounded_gate_values(self):
        gate = self.module.JointSpatiotemporalGate(
            channels=3, hidden_dim=4, initial_gate_strength=0.25
        )
        encoded = torch.randn(2, 5, 3, 4, 6)
        ls = torch.linspace(0.0, 180.0, 5).repeat(2, 1)
        topography = torch.randn(2, 1, 8, 12) * 1_000.0

        gated, gate_values, scales = gate(encoded, ls, topography)

        self.assertEqual(gated.shape, encoded.shape)
        self.assertEqual(gate_values.shape, encoded.shape)
        self.assertEqual(scales.shape, encoded.shape)
        self.assertTrue(torch.all(gate_values >= -1.0))
        self.assertTrue(torch.all(gate_values <= 1.0))
        torch.testing.assert_close(gated, scales * encoded)

    def test_joint_gate_scales_stay_within_initial_strength_and_backpropagates(self):
        gate = self.module.JointSpatiotemporalGate(
            channels=6, hidden_dim=8, initial_gate_strength=0.05
        )
        encoded = torch.randn(2, 3, 6, 4, 5)
        ls = torch.tensor([[0.0, 90.0, 180.0], [45.0, 135.0, 225.0]])
        topography = torch.randn(2, 1, 8, 10) * 1_000.0

        gated, _, scales = gate(encoded, ls, topography)
        self.assertGreaterEqual(scales.min().item(), 0.95 - 1e-6)
        self.assertLessEqual(scales.max().item(), 1.05 + 1e-6)

        gated.square().mean().backward()
        for name, parameter in gate.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_joint_gate_rejects_invalid_encoded_rank_and_channels(self):
        gate = self.module.JointSpatiotemporalGate(
            channels=3, hidden_dim=4, initial_gate_strength=0.25
        )
        ls = torch.zeros(2, 5)
        topography = torch.zeros(2, 1, 8, 12)

        with self.assertRaisesRegex(ValueError, "rank-5"):
            gate(torch.zeros(2, 3, 4, 6), ls, topography)
        with self.assertRaisesRegex(ValueError, "channels"):
            gate(torch.zeros(2, 5, 2, 4, 6), ls, topography)

    def test_joint_gate_ls_effect_changes_with_topography(self):
        gate = self.module.JointSpatiotemporalGate(
            channels=1, hidden_dim=1, initial_gate_strength=1.0
        )
        with torch.no_grad():
            gate.feature_projection.weight.zero_()
            gate.feature_projection.bias.zero_()
            gate.ls_encoder.layers[0].weight.zero_()
            gate.ls_encoder.layers[0].bias.zero_()
            gate.ls_encoder.layers[0].weight[0, 0] = 1.0
            gate.ls_encoder.layers[2].weight.fill_(1.0)
            gate.ls_encoder.layers[2].bias.zero_()
            gate.topography_encoder.layers[0].weight.zero_()
            gate.topography_encoder.layers[0].bias.zero_()
            gate.topography_encoder.layers[0].weight[0, 0, 1, 1] = 1.0
            gate.topography_encoder.layers[2].weight.zero_()
            gate.topography_encoder.layers[2].bias.zero_()
            gate.topography_encoder.layers[2].weight[0, 0, 1, 1] = 1.0
            gate.output_projection.weight.fill_(1.0)
            gate.output_projection.bias.zero_()

        encoded = torch.ones(1, 2, 1, 4, 4)
        ls = torch.tensor([[0.0, 90.0]])
        low_terrain = torch.zeros(1, 1, 8, 8)
        high_terrain = torch.full((1, 1, 8, 8), 10_000.0)

        _, low_gate, _ = gate(encoded, ls, low_terrain)
        _, high_gate, _ = gate(encoded, ls, high_terrain)
        low_ls_effect = low_gate[:, 1] - low_gate[:, 0]
        high_ls_effect = high_gate[:, 1] - high_gate[:, 0]

        self.assertFalse(torch.allclose(low_gate, high_gate))
        self.assertFalse(torch.allclose(low_ls_effect, high_ls_effect))

    def test_matches_platform_dry_run_with_both_auxiliary_inputs(self):
        model = self.module.build_model(
            model_config(
                in_channels=1,
                selected_channels=[0],
                window=3,
                horizon=3,
                height=8,
                width=16,
            )
        ).eval()

        with torch.no_grad():
            outputs = model(
                torch.randn(2, 3, 1, 8, 16),
                torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]]),
                torch.randn(2, 1, 8, 16) * 1_000.0,
            )

        self.assertEqual(outputs.shape, (2, 3, 1, 8, 16))

    def test_forward_supports_primary_and_odd_multichannel_shapes(self):
        primary = self.module.build_model(model_config()).eval()
        with torch.no_grad():
            primary_output = primary(
                torch.randn(1, 20, 5, 8, 16),
                torch.linspace(0.0, 190.0, 20).unsqueeze(0),
                torch.linspace(-8_000.0, 16_000.0, 128).reshape(1, 1, 8, 16),
            )
        self.assertEqual(primary_output.shape, (1, 20, 1, 8, 16))

        odd = self.module.build_model(
            model_config(
                in_channels=2,
                selected_channels=[0, 1],
                window=4,
                horizon=2,
                height=9,
                width=15,
                convlstm_hidden_dim=4,
                spatial_hidden_dim=6,
                temporal_hidden_dim=8,
                num_temporal_blocks=1,
                dropout=0.0,
                gate_hidden_dim=8,
            )
        )
        odd_output = odd(
            torch.randn(1, 4, 2, 9, 15),
            torch.tensor([[0.0, 30.0, 60.0, 90.0]]),
            torch.randn(1, 1, 9, 15) * 1_000.0,
        )
        odd_output.square().mean().backward()

        self.assertEqual(odd_output.shape, (1, 2, 1, 9, 15))
        for name, parameter in odd.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_zero_strength_exactly_matches_baseline_with_shared_weights(self):
        joint_config = model_config(
            window=3,
            horizon=2,
            height=8,
            width=16,
            initial_gate_strength=0.0,
        )
        baseline_config = {
            key: value
            for key, value in joint_config.items()
            if key not in {"gate_hidden_dim", "initial_gate_strength"}
        }
        baseline = self.baseline_module.build_model(baseline_config).eval()
        joint = self.module.build_model(joint_config).eval()
        joint_state = joint.state_dict()
        for name, value in baseline.state_dict().items():
            joint_state[name] = value
        joint.load_state_dict(joint_state)

        x = torch.randn(2, 3, 5, 8, 16)
        ls = torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]])
        topography = torch.randn(2, 1, 8, 16) * 2_000.0
        with torch.no_grad():
            expected = baseline(x)
            actual = joint(x, ls, topography)

        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_rejects_invalid_runtime_inputs(self):
        model = self.module.build_model(model_config(window=3, horizon=2))
        x = torch.randn(2, 3, 5, 8, 16)
        valid_ls = torch.zeros(2, 3)
        valid_topography = torch.zeros(2, 1, 8, 16)
        cases = [
            (
                "x",
                lambda: model(
                    torch.randn(2, 3, 5, 8), valid_ls, valid_topography
                ),
            ),
            (
                "window",
                lambda: model(
                    torch.randn(2, 2, 5, 8, 16),
                    valid_ls[:, :2],
                    valid_topography,
                ),
            ),
            (
                "channels",
                lambda: model(
                    torch.randn(2, 3, 4, 8, 16), valid_ls, valid_topography
                ),
            ),
            ("ls", lambda: model(x, None, valid_topography)),
            ("ls", lambda: model(x, torch.zeros(2, 2), valid_topography)),
            (
                "ls",
                lambda: model(
                    x, torch.zeros(2, 3, dtype=torch.long), valid_topography
                ),
            ),
            (
                "ls",
                lambda: model(
                    x,
                    torch.tensor([[0.0, 1.0, float("nan")]]).expand(2, -1),
                    valid_topography,
                ),
            ),
            (
                "device",
                lambda: model(
                    x, torch.zeros(2, 3, device="meta"), valid_topography
                ),
            ),
            ("topography", lambda: model(x, valid_ls, None)),
            ("topography", lambda: model(x, valid_ls, torch.zeros(2, 8, 16))),
            (
                "topography",
                lambda: model(x, valid_ls, torch.zeros(2, 2, 8, 16)),
            ),
            (
                "topography",
                lambda: model(x, valid_ls, torch.zeros(2, 1, 7, 16)),
            ),
            (
                "topography",
                lambda: model(
                    x, valid_ls, torch.zeros(2, 1, 8, 16, dtype=torch.long)
                ),
            ),
            (
                "topography",
                lambda: model(
                    x, valid_ls, torch.full((2, 1, 8, 16), float("inf"))
                ),
            ),
            (
                "device",
                lambda: model(
                    x, valid_ls, torch.zeros(2, 1, 8, 16, device="meta")
                ),
            ),
        ]

        for label, call in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, label):
                    call()

    def test_rejects_invalid_gate_configuration(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.module.build_model(model_config(num_temporal_blocks=0))
        with self.assertRaisesRegex(ValueError, "dropout"):
            self.module.build_model(model_config(dropout=1.0))
        with self.assertRaisesRegex(ValueError, "gate_hidden_dim"):
            self.module.build_model(model_config(gate_hidden_dim=0))
        with self.assertRaisesRegex(ValueError, "initial_gate_strength"):
            self.module.build_model(model_config(initial_gate_strength=-0.01))
        with self.assertRaisesRegex(ValueError, "initial_gate_strength"):
            self.module.build_model(model_config(initial_gate_strength=1.01))

    def test_main_path_classes_are_ast_identical_to_baseline(self):
        class_names = (
            "ConvLSTMCell",
            "ConvLSTMEncoder",
            "SpatialEncoder",
            "TemporalInceptionBlock",
            "TemporalTranslator",
            "SpatialDecoder",
        )
        for class_name in class_names:
            with self.subTest(class_name=class_name):
                self.assertEqual(
                    class_ast(MODEL_PATH, class_name),
                    class_ast(BASELINE_MODEL_PATH, class_name),
                )

    def test_uses_only_upload_safe_imports_and_calls(self):
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
        self.assertLessEqual(import_roots, {"torch"})
        self.assertFalse(called_names & banned_calls)

    def test_primary_configuration_parameter_overhead_is_below_fifteen_percent(self):
        joint_config = model_config()
        baseline_config = {
            key: value
            for key, value in joint_config.items()
            if key not in {"gate_hidden_dim", "initial_gate_strength"}
        }
        joint = self.module.build_model(joint_config)
        baseline = self.baseline_module.build_model(baseline_config)
        baseline_parameters = trainable_parameter_count(baseline)
        overhead = (
            trainable_parameter_count(joint) - baseline_parameters
        ) / baseline_parameters

        self.assertLess(overhead, 0.15)


if __name__ == "__main__":
    unittest.main()
