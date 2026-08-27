import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "convlstm_residual_adaptive_simvp.py"
)


def load_uploaded_model():
    spec = importlib.util.spec_from_file_location(
        "convlstm_residual_adaptive_simvp", MODEL_PATH
    )
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
        "attention_initial_strength": 0.2,
        "residual_branch_scale": 0.05,
    }
    config.update(overrides)
    return config


class ConvLSTMResidualAdaptiveSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_uploaded_model() if MODEL_PATH.exists() else None

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("residual adaptive model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

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
            "attention_initial_strength": {
                "type": "float",
                "default": 0.2,
                "min": 0.01,
                "max": 0.99,
            },
            "residual_branch_scale": {
                "type": "float",
                "default": 0.05,
                "min": 0.0,
                "max": 0.5,
            },
        }

        self.assertEqual(self.module.MODEL_SPEC["parameters"], expected)
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def _encoded_and_baseline(self, model, inputs):
        batch, window, _, height, width = inputs.shape
        recurrent = model.convlstm(inputs)
        encoded_flat = model.spatial_encoder(
            recurrent.reshape(
                batch * window,
                recurrent.shape[2],
                height,
                width,
            )
        )
        encoded_height, encoded_width = encoded_flat.shape[-2:]
        encoded = encoded_flat.reshape(
            batch,
            window,
            model.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        baseline = model.temporal_translator(
            encoded.reshape(
                batch,
                window * model.spatial_hidden_dim,
                encoded_height,
                encoded_width,
            )
        )
        return encoded, baseline

    def test_zero_residual_scale_exactly_preserves_baseline_path(self):
        torch.manual_seed(17)
        model = self.module.build_model(
            model_config(dropout=0.0, residual_branch_scale=0.0)
        ).eval()
        inputs = torch.randn(2, 3, 1, 8, 16)
        encoded, baseline = self._encoded_and_baseline(model, inputs)
        correction = model.adaptive_branch(encoded)
        expected = model.spatial_decoder(
            baseline.reshape(6, 32, 4, 8), (8, 16)
        ).reshape(2, 3, 1, 8, 16)

        actual = model(inputs)

        self.assertEqual(float(correction.abs().max().detach()), 0.0)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_default_attention_is_a_small_trainable_residual(self):
        torch.manual_seed(19)
        model = self.module.build_model(model_config(dropout=0.0)).eval()
        inputs = torch.randn(2, 3, 1, 8, 16)
        encoded, baseline = self._encoded_and_baseline(model, inputs)
        correction = model.adaptive_branch(encoded)
        ratio = correction.norm() / baseline.norm().clamp_min(1e-12)
        weights = model.adaptive_branch.temporal_fusion.attention_weights(encoded)
        outputs = model(inputs)
        query_gradient = torch.autograd.grad(
            outputs.square().mean(),
            model.adaptive_branch.temporal_fusion.forecast_query,
        )[0]

        self.assertGreater(float(correction.norm().detach()), 0.0)
        self.assertLess(float(ratio.detach()), 0.25)
        self.assertGreater(
            float((weights[:, 0] - weights[:, 1]).abs().max().detach()),
            1e-5,
        )
        self.assertTrue(torch.isfinite(query_gradient).all())
        self.assertGreater(float(query_gradient.abs().sum().detach()), 1e-12)

    def test_matches_platform_dry_run_shape(self):
        model = self.module.build_model(model_config()).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 1, 8, 16))

        self.assertEqual(tuple(outputs.shape), (2, 3, 1, 8, 16))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_supports_odd_multichannel_inputs_and_full_backpropagation(self):
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
            attention_initial_strength=0.3,
            residual_branch_scale=0.05,
        )
        model = self.module.build_model(config)
        outputs = model(torch.randn(1, 4, 2, 9, 15))
        outputs.square().mean().backward()

        self.assertEqual(tuple(outputs.shape), (1, 2, 1, 9, 15))
        self.assertTrue(torch.isfinite(outputs).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing gradient: {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
            )

    def test_rejects_invalid_configuration_and_runtime_shapes(self):
        with self.assertRaisesRegex(ValueError, "window"):
            self.module.build_model(model_config(window=1))
        with self.assertRaisesRegex(ValueError, "residual_branch_scale"):
            self.module.build_model(model_config(residual_branch_scale=0.6))
        with self.assertRaisesRegex(ValueError, "attention_initial_strength"):
            self.module.build_model(model_config(attention_initial_strength=0.0))

        model = self.module.build_model(model_config())
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 3, 8, 16))
        with self.assertRaisesRegex(ValueError, "window"):
            model(torch.randn(1, 2, 1, 8, 16))
        with self.assertRaisesRegex(ValueError, "channels"):
            model(torch.randn(1, 3, 2, 8, 16))

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
        self.assertLessEqual(import_roots, {"torch", "numpy"})
        self.assertFalse(called_names & banned_calls)


if __name__ == "__main__":
    unittest.main()
