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

    def test_future_ls_continues_across_degree_wrap(self):
        encoder = self.module.FutureLsEncoder()
        ls = torch.tensor([[358.0, 359.0, 0.0, 1.0]])
        angles = encoder.future_angles(ls, horizon=3)
        expected = torch.deg2rad(torch.tensor([[2.0, 3.0, 4.0]]))
        torch.testing.assert_close(angles, expected, atol=1e-5, rtol=0.0)

    def test_future_ls_harmonics_have_four_orders(self):
        encoder = self.module.FutureLsEncoder()
        harmonics = encoder.harmonics(torch.tensor([[0.0, torch.pi / 2]]))
        self.assertEqual(harmonics.shape, (1, 2, 8))
        torch.testing.assert_close(
            harmonics[0, 0],
            torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_bilinear_weights_normalize_over_neighbors_and_backpropagate(self):
        module = self.module.SeasonalTerrainWeights(
            edge_dim=43,
            terrain_hidden_dim=8,
            heads=4,
        )
        edges = torch.randn(2, 24, 43, 8, 16)
        season = torch.randn(2, 8)
        encoded_edges = module.encode_edges(edges)
        weights = module(encoded_edges, season)
        self.assertEqual(weights.shape, (2, 24, 4, 8, 16))
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.ones(2, 4, 8, 16),
            atol=1e-6,
            rtol=1e-6,
        )
        (weights.square().mean() + encoded_edges.square().mean()).backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_bilinear_terrain_response_changes_with_season(self):
        module = self.module.SeasonalTerrainWeights(
            edge_dim=43,
            terrain_hidden_dim=8,
            heads=2,
        )
        edges = torch.randn(1, 24, 43, 4, 8)
        encoded_edges = module.encode_edges(edges)
        season_a = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
        )
        season_b = torch.tensor(
            [[1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0]]
        )
        weights_a = module(encoded_edges, season_a)
        weights_b = module(encoded_edges, season_b)
        self.assertGreater((weights_a - weights_b).abs().max().item(), 1e-7)

    def test_history_encoder_preserves_full_resolution(self):
        encoder = self.module.HistoryEncoder(hidden_dim=16)
        inputs = torch.randn(2, 4, 5, 9, 16)
        hidden, context = encoder(inputs)
        self.assertEqual(hidden.shape, (2, 16, 9, 16))
        self.assertEqual(context.shape, hidden.shape)

    def test_history_encoder_has_separate_ozone_and_forcing_stems(self):
        encoder = self.module.HistoryEncoder(hidden_dim=16)
        self.assertEqual(encoder.ozone_stem.in_channels, 1)
        self.assertEqual(encoder.forcing_stem.in_channels, 4)

    def test_history_encoder_all_input_channels_receive_gradients(self):
        encoder = self.module.HistoryEncoder(hidden_dim=16)
        inputs = torch.randn(2, 3, 5, 8, 16, requires_grad=True)
        hidden, context = encoder(inputs)
        (hidden.square().mean() + context.square().mean()).backward()
        for channel in range(5):
            self.assertGreater(
                inputs.grad[:, :, channel].abs().sum().item(),
                0.0,
            )

    def test_steo_returns_state_shape_and_normalized_weights(self):
        operator = self.module.STEO(
            hidden_dim=16,
            edge_dim=43,
            terrain_hidden_dim=8,
            heads=4,
        )
        state = torch.randn(2, 16, 8, 16)
        edges = torch.randn(2, 24, 43, 8, 16)
        season = torch.randn(2, 8)
        encoded_edges = operator.encode_edges(edges)
        update, weights = operator(state, encoded_edges, season)
        self.assertEqual(update.shape, state.shape)
        self.assertEqual(weights.shape, (2, 24, 4, 8, 16))
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.ones(2, 4, 8, 16),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_steo_constant_state_has_exact_zero_difference_update(self):
        operator = self.module.STEO(
            hidden_dim=16,
            edge_dim=43,
            terrain_hidden_dim=8,
            heads=4,
        ).eval()
        state = torch.full((2, 16, 8, 16), 3.25)
        edges = torch.randn(2, 24, 43, 8, 16)
        season = torch.randn(2, 8)
        encoded_edges = operator.encode_edges(edges)
        update, _ = operator(state, encoded_edges, season)
        torch.testing.assert_close(
            update,
            torch.zeros_like(update),
            atol=0.0,
            rtol=0.0,
        )

    def test_steo_backpropagates_to_state_edges_and_all_parameters(self):
        operator = self.module.STEO(
            hidden_dim=16,
            edge_dim=43,
            terrain_hidden_dim=8,
            heads=4,
        )
        state = torch.randn(2, 16, 8, 16, requires_grad=True)
        edges = torch.randn(2, 24, 43, 8, 16, requires_grad=True)
        encoded_edges = operator.encode_edges(edges)
        update, weights = operator(state, encoded_edges, torch.randn(2, 8))
        (update.square().mean() + weights.square().mean()).backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(edges.grad).all())
        for name, parameter in operator.named_parameters():
            self.assertIsNotNone(parameter.grad, name)


if __name__ == "__main__":
    unittest.main()
