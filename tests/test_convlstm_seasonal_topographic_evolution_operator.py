import ast
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "models" / "convlstm_seasonal_topographic_evolution_operator.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("st_topographic_operator", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_config(**overrides):
    config = {
        "in_channels": 5,
        "window": 20,
        "horizon": 20,
        "height": 36,
        "width": 72,
        "selected_channels": [0, 1, 2, 3, 4],
        "history_hidden_dim": 32,
        "terrain_hidden_dim": 24,
        "operator_heads": 4,
        "evolution_blocks": 2,
        "dropout": 0.1,
    }
    config.update(overrides)
    return config


class SeasonalTopographicEvolutionOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module() if MODEL_PATH.exists() else None

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("STEO model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

    def test_exports_exact_contract(self):
        self.assertTrue(hasattr(self.module, "MODEL_SPEC"))
        self.assertTrue(callable(self.module.build_model))
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
        parameters = inspect.signature(
            self.module.SeasonalTopographicEvolutionOperator.forward
        ).parameters
        self.assertEqual(list(parameters), ["self", "x", "ls", "topography"])

    def test_builder_returns_module_and_rejects_non_five_channel_config(self):
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)
        with self.assertRaisesRegex(ValueError, "in_channels must be exactly 5"):
            self.module.build_model(model_config(in_channels=4))


if __name__ == "__main__":
    unittest.main()
