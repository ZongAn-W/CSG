import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "convlstm_adaptive_simvp.py"
)


def load_uploaded_model():
    spec = importlib.util.spec_from_file_location(
        "convlstm_adaptive_simvp", MODEL_PATH
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
    }
    config.update(overrides)
    return config


class ConvLSTMAdaptiveSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_uploaded_model() if MODEL_PATH.exists() else None

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("adaptive model file does not exist yet")

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
        }

        self.assertEqual(self.module.MODEL_SPEC["parameters"], expected)
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_attention_is_normalized_channel_and_horizon_specific(self):
        torch.manual_seed(7)
        fusion = self.module.ChannelTemporalFusion(
            window=4,
            horizon=2,
            channels=6,
            initial_strength=0.2,
        )
        encoded = torch.randn(2, 4, 6, 5, 7, requires_grad=True)

        weights = fusion.attention_weights(encoded)
        probe = weights[:, 0, 0, :].square().sum()
        gradient = torch.autograd.grad(probe, encoded)[0]

        self.assertEqual(tuple(weights.shape), (2, 2, 4, 6))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=2),
                torch.ones(2, 2, 6),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(
                torch.sigmoid(fusion.mix_logit),
                torch.full((6,), 0.2),
                atol=1e-6,
            )
        )

    def test_attention_is_materially_adaptive_and_trains_forecast_queries(self):
        torch.manual_seed(11)
        model = self.module.build_model(model_config(dropout=0.0))
        inputs = torch.randn(2, 3, 1, 8, 16)
        recurrent = model.convlstm(inputs)
        encoded = model.spatial_encoder(
            recurrent.reshape(6, 16, 8, 16)
        ).reshape(2, 3, 32, 4, 8)

        weights = model.temporal_fusion.attention_weights(encoded)
        sample_delta = (weights[0] - weights[1]).abs().max()
        horizon_delta = (weights[:, 0] - weights[:, 1]).abs().max()
        channel_delta = (weights[..., 0] - weights[..., 1]).abs().max()
        outputs = model(inputs)
        query_gradient = torch.autograd.grad(
            outputs.square().mean(), model.temporal_fusion.forecast_query
        )[0]

        self.assertGreater(float(sample_delta.detach()), 1e-6)
        self.assertGreater(float(horizon_delta.detach()), 1e-5)
        self.assertGreater(float(channel_delta.detach()), 1e-6)
        self.assertGreater(float(query_gradient.abs().sum().detach()), 1e-8)
        self.assertIsNone(model.temporal_fusion.score_projection.bias)

    def test_matches_platform_dry_run_shape(self):
        model = self.module.build_model(model_config()).eval()
        inputs = torch.randn(2, 3, 1, 8, 16)

        with torch.no_grad():
            outputs = model(inputs)

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
        with self.assertRaisesRegex(ValueError, "attention_initial_strength"):
            self.module.build_model(model_config(attention_initial_strength=1.0))
        with self.assertRaisesRegex(ValueError, "dropout"):
            self.module.build_model(model_config(dropout=-0.1))

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
