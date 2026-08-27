import ast
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "interleaved_convlstm_simvp.py"
)


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "interleaved_convlstm_simvp", MODEL_PATH
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
    }
    config.update(overrides)
    return config


class InterleavedConvLSTMSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_model_module()

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

    def test_exports_platform_model_contract(self):
        expected_parameters = {
            "convlstm_hidden_dim",
            "spatial_hidden_dim",
            "temporal_hidden_dim",
            "num_temporal_blocks",
            "dropout",
        }

        self.assertEqual(
            set(self.module.MODEL_SPEC["parameters"]), expected_parameters
        )
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)

    def test_each_simvp_block_contains_and_executes_a_convlstm(self):
        model = self.module.build_model(model_config(num_temporal_blocks=2)).eval()
        calls = [0, 0]
        hooks = [
            block.convlstm.register_forward_hook(
                lambda _module, _inputs, _output, index=index: calls.__setitem__(
                    index, calls[index] + 1
                )
            )
            for index, block in enumerate(model.temporal_translator.blocks)
        ]

        try:
            with torch.no_grad():
                outputs = model(torch.randn(2, 3, 1, 8, 16))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(len(model.temporal_translator.blocks), 2)
        self.assertFalse(hasattr(model, "convlstm"))
        self.assertEqual(calls, [1, 1])
        self.assertEqual(tuple(outputs.shape), (2, 3, 1, 8, 16))

    def test_matches_platform_dry_run_shape(self):
        model = self.module.build_model(model_config()).eval()
        inputs = torch.randn(2, 3, 1, 8, 16)

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (2, 3, 1, 8, 16))

    def test_supports_multichannel_odd_sized_inputs_and_backpropagation(self):
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
            num_temporal_blocks=2,
            dropout=0.0,
        )
        model = self.module.build_model(config)
        inputs = torch.randn(1, 4, 2, 9, 15)

        outputs = model(inputs)
        outputs.square().mean().backward()

        self.assertEqual(tuple(outputs.shape), (1, 2, 1, 9, 15))
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_rejects_runtime_window_mismatch(self):
        model = self.module.build_model(model_config())

        with self.assertRaisesRegex(ValueError, "window"):
            model(torch.randn(1, 2, 1, 8, 16))

    def test_rejects_invalid_build_parameters(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            self.module.build_model(model_config(dropout=1.0))

        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.module.build_model(model_config(num_temporal_blocks=0))

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
