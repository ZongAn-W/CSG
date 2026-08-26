import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "stacked_spatiotemporal_residual_net.py"
)


def load_uploaded_model():
    spec = importlib.util.spec_from_file_location(
        "stacked_spatiotemporal_residual_net", MODEL_PATH
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
        "hidden_dim": 32,
        "spatial_dim": 128,
        "num_blocks": 3,
        "dropout": 0.1,
    }
    config.update(overrides)
    return config


class StackedSpatiotemporalResidualNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_uploaded_model() if MODEL_PATH.exists() else None

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

    def test_exports_platform_model_contract(self):
        expected_parameters = {
            "hidden_dim": {
                "type": "int",
                "default": 32,
                "min": 4,
                "max": 128,
            },
            "spatial_dim": {
                "type": "int",
                "default": 128,
                "min": 8,
                "max": 512,
            },
            "num_blocks": {
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

        self.assertEqual(self.module.MODEL_SPEC["parameters"], expected_parameters)
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_block_matches_approved_architecture(self):
        block = self.module.SpatiotemporalResidualBlock(
            window=3, hidden_dim=4, spatial_dim=8, dropout=0.0
        )

        self.assertEqual(len(block.temporal.layers), 2)
        self.assertEqual(block.branch_3x3.kernel_size, (3, 3))
        self.assertEqual(block.branch_5x5.kernel_size, (5, 5))
        self.assertEqual(block.branch_7x7.kernel_size, (7, 7))
        self.assertEqual(block.branch_3x3.groups, 1)
        outputs = block(torch.randn(2, 3, 4, 7, 9))
        self.assertEqual(tuple(outputs.shape), (2, 3, 4, 7, 9))

    def test_zero_updates_preserve_block_input(self):
        block = self.module.SpatiotemporalResidualBlock(
            window=3, hidden_dim=4, spatial_dim=8, dropout=0.0
        ).eval()
        inputs = torch.randn(2, 3, 4, 7, 9)
        for parameter in block.temporal.parameters():
            nn.init.zeros_(parameter)
        nn.init.zeros_(block.output_projection.weight)
        nn.init.zeros_(block.output_projection.bias)

        self.assertTrue(torch.equal(block(inputs), inputs))

    def test_matches_platform_dry_run_shape(self):
        model = self.module.build_model(model_config()).eval()
        self.assertEqual(len(model.blocks), 3)
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 1, 8, 16))

        self.assertEqual(tuple(outputs.shape), (2, 3, 1, 8, 16))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_supports_general_time_channel_and_spatial_shapes(self):
        model = self.module.build_model(
            model_config(
                in_channels=2,
                window=4,
                horizon=2,
                selected_channels=[0, 1],
                hidden_dim=4,
                spatial_dim=8,
                num_blocks=2,
                dropout=0.0,
            )
        )
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
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.module.build_model(model_config(num_blocks=0))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.module.build_model(model_config(hidden_dim=True))
        with self.assertRaisesRegex(ValueError, "dropout"):
            self.module.build_model(model_config(dropout=1.0))

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
        self.assertLessEqual(import_roots, {"torch"})
        self.assertFalse(called_names & banned_calls)


if __name__ == "__main__":
    unittest.main()
