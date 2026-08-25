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


if __name__ == "__main__":
    unittest.main()
