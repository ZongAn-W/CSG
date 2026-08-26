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
        "operator_mode": "bilinear",
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

    def test_primary_model_forecasts_twenty_full_resolution_frames(self):
        model = self.module.build_model(model_config()).eval()
        x = torch.randn(1, 20, 5, 36, 72)
        ls = torch.linspace(350.0, 369.0, 20).remainder(360.0).unsqueeze(0)
        topography = torch.randn(1, 1, 36, 72) * 2000.0
        with torch.no_grad():
            output = model(x, ls, topography)
        self.assertEqual(output.shape, (1, 20, 1, 36, 72))
        self.assertTrue(torch.isfinite(output).all())

    def test_platform_dry_run_shape_and_finite_backward(self):
        model = self.module.build_model(
            model_config(
                window=3,
                horizon=2,
                height=8,
                width=16,
                history_hidden_dim=8,
                terrain_hidden_dim=8,
                operator_heads=2,
                evolution_blocks=1,
                dropout=0.0,
            )
        )
        x = torch.randn(2, 3, 5, 8, 16)
        ls = torch.tensor([[358.0, 359.0, 0.0], [45.0, 46.0, 47.0]])
        topography = torch.randn(2, 1, 8, 16) * 1000.0
        output = model(x, ls, topography)
        output.square().mean().backward()
        self.assertEqual(output.shape, (2, 2, 1, 8, 16))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_diagnostics_expose_lead_block_neighbor_head_weights(self):
        model = self.module.build_model(
            model_config(
                window=3,
                horizon=2,
                height=8,
                width=16,
                history_hidden_dim=8,
                terrain_hidden_dim=8,
                operator_heads=2,
                evolution_blocks=1,
                dropout=0.0,
            )
        ).eval()
        x = torch.randn(1, 3, 5, 8, 16)
        ls = torch.tensor([[10.0, 11.0, 12.0]])
        topography = torch.randn(1, 1, 8, 16) * 1000.0
        with torch.no_grad():
            output, weights = model.forward_with_diagnostics(x, ls, topography)
        self.assertEqual(output.shape, (1, 2, 1, 8, 16))
        self.assertEqual(weights.shape, (1, 2, 1, 24, 2, 8, 16))

    def test_rejects_invalid_runtime_inputs(self):
        model = self.module.build_model(
            model_config(
                window=3,
                horizon=2,
                history_hidden_dim=8,
                terrain_hidden_dim=8,
                operator_heads=2,
                evolution_blocks=1,
            )
        )
        x = torch.randn(2, 3, 5, 8, 16)
        ls = torch.zeros(2, 3)
        topography = torch.zeros(2, 1, 8, 16)
        cases = [
            ("x", lambda: model(torch.randn(2, 3, 5, 8), ls, topography)),
            (
                "window",
                lambda: model(
                    torch.randn(2, 2, 5, 8, 16),
                    ls[:, :2],
                    topography,
                ),
            ),
            (
                "channels",
                lambda: model(torch.randn(2, 3, 4, 8, 16), ls, topography),
            ),
            (
                "finite",
                lambda: model(torch.full_like(x, float("nan")), ls, topography),
            ),
            (
                "grid",
                lambda: model(
                    torch.randn(2, 3, 5, 7, 16),
                    ls,
                    torch.zeros(2, 1, 7, 16),
                ),
            ),
            (
                "grid",
                lambda: model(
                    torch.randn(2, 3, 5, 8, 15),
                    ls,
                    torch.zeros(2, 1, 8, 15),
                ),
            ),
            ("ls", lambda: model(x, torch.zeros(2, 2), topography)),
            (
                "ls",
                lambda: model(x, torch.zeros(2, 3, dtype=torch.long), topography),
            ),
            (
                "ls",
                lambda: model(x, torch.full((2, 3), float("inf")), topography),
            ),
            (
                "device",
                lambda: model(x, torch.empty(2, 3, device="meta"), topography),
            ),
            ("topography", lambda: model(x, ls, torch.zeros(2, 8, 16))),
            ("float32", lambda: model(x, ls, topography.double())),
            (
                "topography",
                lambda: model(
                    x,
                    ls,
                    torch.full_like(topography, float("nan")),
                ),
            ),
            (
                "device",
                lambda: model(
                    x,
                    ls,
                    torch.empty(2, 1, 8, 16, device="meta"),
                ),
            ),
        ]
        for label, call in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, label):
                    call()

    def test_rejects_invalid_configuration(self):
        invalid = [
            ({"window": 1}, "window"),
            ({"history_hidden_dim": 0}, "positive integer"),
            ({"terrain_hidden_dim": 0}, "positive integer"),
            ({"operator_heads": 0}, "positive integer"),
            ({"evolution_blocks": 0}, "positive integer"),
            ({"history_hidden_dim": 10, "operator_heads": 4}, "divisible"),
            ({"dropout": 1.0}, "dropout"),
            ({"dropout": -0.1}, "dropout"),
        ]
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.module.build_model(model_config(**overrides))

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
        banned = {
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
        self.assertFalse(called_names & banned)

    def test_future_non_operator_modules_are_pointwise(self):
        model = self.module.build_model(model_config())
        for block_index, block in enumerate(model.blocks):
            for branch_name in ("pointwise", "step_projection"):
                branch = getattr(block, branch_name)
                for module in branch.modules():
                    if isinstance(module, nn.Conv2d):
                        self.assertEqual(
                            module.kernel_size,
                            (1, 1),
                            f"block {block_index} {branch_name}",
                        )
        for module in model.output_head.modules():
            if isinstance(module, nn.Conv2d):
                self.assertEqual(module.kernel_size, (1, 1))

    def test_model_has_no_trainable_per_pixel_parameters(self):
        model = self.module.build_model(model_config())
        forbidden_spatial_shapes = {(36, 72), (1, 36, 72)}
        for name, parameter in model.named_parameters():
            self.assertNotIn(tuple(parameter.shape), forbidden_spatial_shapes, name)
            self.assertNotEqual(tuple(parameter.shape[-2:]), (36, 72), name)

    def test_eval_is_deterministic_and_reports_parameter_count(self):
        model = self.module.build_model(
            model_config(
                window=3,
                horizon=2,
                height=8,
                width=16,
                history_hidden_dim=8,
                terrain_hidden_dim=8,
                operator_heads=2,
                evolution_blocks=1,
            )
        ).eval()
        x = torch.randn(1, 3, 5, 8, 16)
        ls = torch.tensor([[10.0, 11.0, 12.0]])
        topography = torch.randn(1, 1, 8, 16) * 1000.0
        with torch.no_grad():
            first = model(x, ls, topography)
            second = model(x, ls, topography)
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertGreater(parameter_count, 0)
        print(f"STEO dry-run trainable parameters: {parameter_count}")

    def test_schema_exposes_exact_operator_ablation_modes(self):
        parameter = self.module.MODEL_SPEC["parameters"]["operator_mode"]
        self.assertEqual(parameter["type"], "select")
        self.assertEqual(parameter["default"], "bilinear")
        self.assertEqual(
            parameter["options"],
            [
                "fixed",
                "terrain_only",
                "ls_only",
                "separable",
                "bilinear",
                "ordinary_conv",
            ],
        )

    def test_all_operator_modes_complete_the_same_dry_run(self):
        x = torch.randn(1, 3, 5, 8, 16)
        ls = torch.tensor([[20.0, 21.0, 22.0]])
        topography = torch.randn(1, 1, 8, 16) * 1000.0
        modes = (
            "fixed",
            "terrain_only",
            "ls_only",
            "separable",
            "bilinear",
            "ordinary_conv",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                model = self.module.build_model(
                    model_config(
                        window=3,
                        horizon=2,
                        height=8,
                        width=16,
                        history_hidden_dim=8,
                        terrain_hidden_dim=8,
                        operator_heads=2,
                        evolution_blocks=1,
                        dropout=0.0,
                        operator_mode=mode,
                    )
                ).eval()
                with torch.no_grad():
                    output = model(x, ls, topography)
                self.assertEqual(output.shape, (1, 2, 1, 8, 16))
                self.assertTrue(torch.isfinite(output).all())

    def test_ablation_dependencies_match_their_names(self):
        edges_a = torch.randn(1, 24, 43, 4, 8)
        edges_b = torch.randn(1, 24, 43, 4, 8)
        season_a = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
        )
        season_b = torch.tensor(
            [[1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0]]
        )

        fixed = self.module.SeasonalTerrainWeights(43, 8, 2, mode="fixed")
        fixed_a = fixed(fixed.encode_edges(edges_a), season_a)
        fixed_b = fixed(fixed.encode_edges(edges_b), season_b)
        torch.testing.assert_close(fixed_a, fixed_b)

        terrain = self.module.SeasonalTerrainWeights(
            43,
            8,
            2,
            mode="terrain_only",
        )
        terrain_a = terrain(terrain.encode_edges(edges_a), season_a)
        terrain_b = terrain(terrain.encode_edges(edges_a), season_b)
        torch.testing.assert_close(terrain_a, terrain_b)
        self.assertGreater(
            (
                terrain_a
                - terrain(terrain.encode_edges(edges_b), season_a)
            ).abs().max().item(),
            1e-7,
        )

        ls_only = self.module.SeasonalTerrainWeights(
            43,
            8,
            2,
            mode="ls_only",
        )
        ls_edges_a = ls_only.encode_edges(edges_a)
        ls_edges_b = ls_only.encode_edges(edges_b)
        torch.testing.assert_close(
            ls_only(ls_edges_a, season_a),
            ls_only(ls_edges_b, season_a),
        )
        self.assertGreater(
            (
                ls_only(ls_edges_a, season_a)
                - ls_only(ls_edges_a, season_b)
            ).abs().max().item(),
            1e-7,
        )

    def test_ordinary_ablation_uses_spatial_convolution_block(self):
        model = self.module.build_model(model_config(operator_mode="ordinary_conv"))
        self.assertIsInstance(
            model.blocks[0],
            self.module.OrdinaryEvolutionBlock,
        )

    def test_nonseasonal_modes_zero_the_future_season_context(self):
        harmonics = torch.randn(2, 3, 8)
        for mode in ("fixed", "terrain_only"):
            model = self.module.build_model(model_config(operator_mode=mode))
            conditioned = model._condition_harmonics(harmonics)
            torch.testing.assert_close(conditioned, torch.zeros_like(harmonics))
        for mode in (
            "ls_only",
            "separable",
            "bilinear",
            "ordinary_conv",
        ):
            model = self.module.build_model(model_config(operator_mode=mode))
            torch.testing.assert_close(
                model._condition_harmonics(harmonics),
                harmonics,
            )


if __name__ == "__main__":
    unittest.main()
