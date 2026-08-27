import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "convlstm_state_motion_simvp.py"
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
    }
    config.update(overrides)
    return config


class ConvLSTMStateMotionSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (
            load_module("convlstm_state_motion_simvp", MODEL_PATH)
            if MODEL_PATH.exists()
            else None
        )
        cls.baseline_module = load_module(
            "convlstm_simvp_baseline", BASELINE_MODEL_PATH
        )

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("state-motion model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

    def test_matches_platform_dry_run_shape(self):
        model = self.module.build_model(model_config()).eval()
        inputs = torch.randn(2, 3, 1, 8, 16)

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 3, 1, 8, 16))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_supports_multichannel_different_horizon_and_odd_spatial_size(self):
        config = model_config(
            in_channels=2,
            window=4,
            horizon=2,
            height=9,
            width=15,
            selected_channels=[0, 1],
            convlstm_hidden_dim=4,
            spatial_hidden_dim=6,
            temporal_hidden_dim=8,
            num_temporal_blocks=1,
            dropout=0.0,
        )
        model = self.module.build_model(config).eval()

        with torch.no_grad():
            outputs = model(torch.randn(1, 4, 2, 9, 15))

        self.assertEqual(tuple(outputs.shape), (1, 2, 1, 9, 15))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_motion_is_zero_then_adjacent_state_difference(self):
        state = torch.tensor(
            [[[[[1.0]]], [[[4.0]]], [[[-2.0]]], [[[3.0]]]]]
        )

        motion = self.module._build_motion(state)

        expected = torch.tensor(
            [[[[[0.0]]], [[[3.0]]], [[[-6.0]]], [[[5.0]]]]]
        )
        self.assertEqual(tuple(motion.shape), tuple(state.shape))
        self.assertTrue(torch.equal(motion, expected))

    def test_single_frame_window_has_zero_motion(self):
        state = torch.randn(2, 1, 3, 4, 5)
        motion = self.module._build_motion(state)
        model = self.module.build_model(
            model_config(window=1, horizon=2, spatial_hidden_dim=4)
        ).eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 1, 1, 7, 9))

        self.assertTrue(torch.equal(motion, torch.zeros_like(state)))
        self.assertEqual(tuple(outputs.shape), (2, 2, 1, 7, 9))

    def test_motion_projection_starts_strictly_zero(self):
        model = self.module.build_model(model_config())
        weight = model.temporal_translator.motion_projection

        self.assertIsInstance(weight, nn.Parameter)
        self.assertEqual(tuple(weight.shape), (64, 3 * 32, 1, 1))
        self.assertEqual(torch.count_nonzero(weight).item(), 0)

    def test_motion_projection_gets_finite_nonzero_gradient_on_first_backward(self):
        torch.manual_seed(23)
        model = self.module.build_model(model_config(dropout=0.0))
        outputs = model(torch.randn(2, 3, 1, 8, 16))

        outputs.square().mean().backward()

        gradient = model.temporal_translator.motion_projection.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_zero_motion_path_is_elementwise_identical_to_c_s_1(self):
        torch.manual_seed(29)
        config = model_config(dropout=0.0)
        baseline = self.baseline_module.build_model(config).eval()
        state_motion = self.module.build_model(config).eval()

        state_motion.convlstm.load_state_dict(baseline.convlstm.state_dict())
        state_motion.spatial_encoder.load_state_dict(
            baseline.spatial_encoder.state_dict()
        )
        state_motion.temporal_translator.state_projection.load_state_dict(
            baseline.temporal_translator.input_projection.state_dict()
        )
        state_motion.temporal_translator.blocks.load_state_dict(
            baseline.temporal_translator.blocks.state_dict()
        )
        state_motion.temporal_translator.output_projection.load_state_dict(
            baseline.temporal_translator.output_projection.state_dict()
        )
        state_motion.spatial_decoder.load_state_dict(
            baseline.spatial_decoder.state_dict()
        )

        inputs = torch.randn(2, 3, 1, 9, 15)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            state_motion_outputs = state_motion(inputs)

        self.assertTrue(torch.equal(state_motion_outputs, baseline_outputs))

    def test_all_trainable_parameters_get_finite_gradients(self):
        torch.manual_seed(31)
        config = model_config(
            in_channels=2,
            window=4,
            horizon=2,
            convlstm_hidden_dim=4,
            spatial_hidden_dim=6,
            temporal_hidden_dim=8,
            num_temporal_blocks=1,
            dropout=0.0,
        )
        model = self.module.build_model(config)
        outputs = model(torch.randn(2, 4, 2, 9, 15))
        outputs.square().mean().backward()

        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing gradient: {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                f"non-finite gradient: {name}",
            )

    def test_exports_complete_platform_parameter_schema(self):
        expected = {
            "convlstm_hidden_dim": {
                "type": "int",
                "default": 16,
                "min": 4,
                "max": 128,
            },
            "spatial_hidden_dim": {
                "type": "int",
                "default": 32,
                "min": 4,
                "max": 256,
            },
            "temporal_hidden_dim": {
                "type": "int",
                "default": 64,
                "min": 8,
                "max": 512,
            },
            "num_temporal_blocks": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 8,
            },
            "dropout": {
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 0.9,
            },
        }

        self.assertEqual(self.module.MODEL_SPEC["parameters"], expected)
        self.assertIsInstance(self.module.MODEL_SPEC["name"], str)
        self.assertTrue(self.module.MODEL_SPEC["name"])
        self.assertIsInstance(self.module.MODEL_SPEC["description"], str)
        self.assertTrue(self.module.MODEL_SPEC["description"])
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_rejects_invalid_configuration_and_runtime_shapes(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.module.build_model(model_config(window=0))
        with self.assertRaisesRegex(ValueError, "dropout"):
            self.module.build_model(model_config(dropout=1.0))

        model = self.module.build_model(model_config())
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 3, 8, 16))
        with self.assertRaisesRegex(ValueError, "window"):
            model(torch.randn(1, 2, 1, 8, 16))
        with self.assertRaisesRegex(ValueError, "channels"):
            model(torch.randn(1, 3, 2, 8, 16))

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
