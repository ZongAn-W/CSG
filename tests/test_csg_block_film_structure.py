import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "CSG_block_film.py"


def load_module():
    spec = importlib.util.spec_from_file_location("csg_block_film", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(**overrides):
    values = {
        "in_channels": 1,
        "window": 3,
        "horizon": 2,
        "convlstm_hidden_dim": 4,
        "spatial_hidden_dim": 4,
        "temporal_hidden_dim": 8,
        "num_temporal_blocks": 2,
        "dropout": 0.0,
        "condition_hidden_dim": 4,
        "initial_gate_strength": 0.05,
        "film_strength": 0.0,
    }
    values.update(overrides)
    return values


class CSGBlockFilmStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_joint_gate_is_per_step_and_feature_dependent(self):
        gate = self.module.JointSpatiotemporalGate(
            channels=2, hidden_dim=4, initial_gate_strength=0.05
        )
        encoded = torch.randn(2, 3, 2, 4, 5)
        ls = torch.tensor([[0.0, 90.0, 180.0], [30.0, 120.0, 210.0]])
        topography = torch.randn(2, 1, 8, 10)

        _, first_gate, _ = gate(encoded, ls, topography)
        _, permuted_gate, _ = gate(encoded, ls[:, [2, 1, 0]], topography)
        _, changed_gate, _ = gate(encoded + 0.5, ls, topography)

        self.assertEqual(first_gate.shape, encoded.shape)
        self.assertFalse(torch.allclose(first_gate, permuted_gate))
        self.assertFalse(torch.allclose(first_gate, changed_gate))

    def test_default_film_is_identity_preserving(self):
        model = self.module.build_model(config()).eval()
        self.assertEqual(model.film_strength, 0.0)

        x = torch.randn(2, 3, 1, 8, 8)
        ls = torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]])
        topography = torch.randn(2, 1, 8, 8)
        with torch.no_grad():
            output = model(x, ls, topography)

        self.assertEqual(output.shape, (2, 2, 1, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_default_model_matches_original_with_same_initialization(self):
        original_spec = importlib.util.spec_from_file_location(
            "csg_original", ROOT / "models" / "CSG.py"
        )
        original = importlib.util.module_from_spec(original_spec)
        original_spec.loader.exec_module(original)

        torch.manual_seed(17)
        original_model = original.build_model(
            {
                **config(),
                "gate_hidden_dim": 4,
            }
        ).eval()
        torch.manual_seed(17)
        corrected_model = self.module.build_model(config()).eval()

        x = torch.randn(2, 3, 1, 8, 8)
        ls = torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]])
        topography = torch.randn(2, 1, 8, 8)
        with torch.no_grad():
            expected = original_model(x, ls, topography)
            actual = corrected_model(x, ls, topography)

        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_enabled_film_remains_bounded_for_large_condition(self):
        block = self.module.TemporalInceptionBlock(
            channels=4, dropout=0.0, condition_hidden_dim=4, film_strength=0.05
        ).eval()
        x = torch.randn(2, 4, 4, 4)
        condition = torch.full((2, 4, 4, 4), 1e6)
        with torch.no_grad():
            output = block(x, condition)

        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(float(output.abs().max()), 1e4)


if __name__ == "__main__":
    unittest.main()
