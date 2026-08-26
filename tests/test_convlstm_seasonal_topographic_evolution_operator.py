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

    def test_spherical_neighbor_wraps_longitude(self):
        grid = self.module.SphericalGrid()
        values = torch.arange(16.0).reshape(1, 1, 2, 8)
        east = grid.neighbor(values, dy=0, dx=1)
        west = grid.neighbor(values, dy=0, dx=-1)
        self.assertEqual(east[0, 0, 0, -1].item(), values[0, 0, 0, 0].item())
        self.assertEqual(west[0, 0, 1, 0].item(), values[0, 0, 1, -1].item())

    def test_spherical_neighbor_reflects_over_pole_with_half_turn(self):
        grid = self.module.SphericalGrid()
        values = torch.arange(32.0).reshape(1, 1, 4, 8)
        north = grid.neighbor(values, dy=-1, dx=0)
        south = grid.neighbor(values, dy=1, dx=0)
        self.assertEqual(north[0, 0, 0, 0].item(), values[0, 0, 0, 4].item())
        self.assertEqual(south[0, 0, -1, 1].item(), values[0, 0, -1, 5].item())

    def test_spherical_distances_contract_toward_poles(self):
        grid = self.module.SphericalGrid()
        east_distance = grid.distance_map(36, 72, dy=0, dx=1, device="cpu")
        north_distance = grid.distance_map(36, 72, dy=1, dx=0, device="cpu")
        self.assertLess(east_distance[0, 0, 0, 0], east_distance[0, 0, 18, 0])
        torch.testing.assert_close(
            north_distance[0, 0, 0, 0], north_distance[0, 0, 18, 0]
        )

    def test_spherical_conv_preserves_shape_and_backpropagates(self):
        layer = self.module.SphericalConv2d(3, 5, kernel_size=3)
        inputs = torch.randn(2, 3, 8, 16, requires_grad=True)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, (2, 5, 8, 16))
        outputs.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_flat_terrain_produces_finite_zero_shape_descriptors(self):
        bank = self.module.TerrainGeometryBank()
        terrain = torch.full((1, 1, 8, 16), 2500.0)
        features = bank(terrain)
        self.assertEqual(features.shape, (1, 12, 8, 16))
        torch.testing.assert_close(features[:, 0], terrain[:, 0] / 10_000.0)
        torch.testing.assert_close(features[:, 1:], torch.zeros_like(features[:, 1:]))
        self.assertTrue(torch.isfinite(features).all())

    def test_eastward_ramp_has_signed_east_slope(self):
        bank = self.module.TerrainGeometryBank()
        column = torch.linspace(-2000.0, 2000.0, 16)
        terrain = column.reshape(1, 1, 1, 16).expand(1, 1, 8, 16).clone()
        features = bank(terrain)
        self.assertGreater(features[0, 1, 4, 8].item(), 0.0)
        self.assertAlmostEqual(features[0, 2, 4, 8].item(), 0.0, places=6)

    def test_bowl_has_center_curvature_distinct_from_flat(self):
        bank = self.module.TerrainGeometryBank()
        yy, xx = torch.meshgrid(
            torch.arange(9.0), torch.arange(16.0), indexing="ij"
        )
        terrain = ((yy - 4.0) ** 2 + (xx - 8.0) ** 2).reshape(1, 1, 9, 16)
        features = bank(terrain)
        self.assertLess(features[0, 6, 4, 8].item(), 0.0)

    def test_edge_builder_creates_twenty_four_relational_neighbors(self):
        bank = self.module.TerrainGeometryBank()
        builder = self.module.TerrainEdgeBuilder()
        terrain = torch.randn(2, 1, 8, 16) * 1000.0
        geometry = bank(terrain)
        edges = builder(geometry)
        self.assertEqual(edges.shape[:2], (2, 24))
        self.assertEqual(edges.shape[-2:], (8, 16))
        self.assertEqual(edges.shape[2], builder.edge_dim)


if __name__ == "__main__":
    unittest.main()
