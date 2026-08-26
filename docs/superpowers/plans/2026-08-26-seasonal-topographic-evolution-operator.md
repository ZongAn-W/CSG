# Seasonal Topographic Evolution Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone AresVision model whose future spatial evolution is performed by a bilinear solar-longitude and terrain-conditioned spherical operator instead of a SimVP temporal translator.

**Architecture:** Separate ozone and meteorological history stems feed a full-resolution ConvLSTM. A deterministic MOLA geometry bank and spherical multiscale edges feed a low-rank `Ls x terrain` weighting network. A shared recurrent core advances the latent state for every forecast lead; all future spatial communication passes through STEO, while the parallel dynamics branch uses pointwise operations only.

**Tech Stack:** Python, PyTorch, `unittest`, `ast`, AresVision single-file uploaded-model contract.

---

## File Structure

- Create `models/convlstm_seasonal_topographic_evolution_operator.py`: complete upload-safe STEO model. AresVision requires a single model file, so all model components live here.
- Create `tests/test_convlstm_seasonal_topographic_evolution_operator.py`: contract, geometry, operator, integration, gradient, and structural tests.
- Reference `models/convlstm_ls_topography_joint_gated_simvp.py`: current strength-1 comparison architecture; do not modify it.
- Reference `tests/test_convlstm_ls_topography_joint_gated_simvp.py`: existing upload-safety and integration-test patterns; do not modify it.
- Reference `docs/superpowers/specs/2026-08-26-seasonal-topographic-evolution-operator-design.md`: approved requirements and scientific claim boundary.

## Task 1: Upload Contract And Builder Skeleton

**Files:**
- Create: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Create: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Write failing contract and builder tests**

Create the test module with the loader, canonical configuration, and initial tests:

```python
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
```

- [ ] **Step 2: Run the tests and confirm the red state**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_model_file_exists tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_exports_exact_contract tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_builder_returns_module_and_rejects_non_five_channel_config -v
```

Expected: `test_model_file_exists` fails because the model file is absent; dependent tests skip.

- [ ] **Step 3: Add the upload metadata, configuration validation, and minimal persistence model**

Create the model file with this initial implementation:

```python
import torch
from torch import nn


MODEL_SPEC = {
    "name": "SeasonalTopographicEvolutionOperator",
    "description": (
        "Full-resolution recurrent Mars ozone model with a bilinear "
        "solar-longitude and terrain-conditioned spherical evolution operator."
    ),
    "parameters": {
        "history_hidden_dim": {"type": "int", "default": 32, "min": 8, "max": 128},
        "terrain_hidden_dim": {"type": "int", "default": 24, "min": 8, "max": 128},
        "operator_heads": {"type": "int", "default": 4, "min": 1, "max": 16},
        "evolution_blocks": {"type": "int", "default": 2, "min": 1, "max": 4},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
    },
    "auxiliary_inputs": {
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
}


def _positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _dropout(config):
    value = config["dropout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= value < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")
    return float(value)


class SeasonalTopographicEvolutionOperator(nn.Module):
    def __init__(
        self,
        in_channels,
        window,
        horizon,
        history_hidden_dim,
        terrain_hidden_dim,
        operator_heads,
        evolution_blocks,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.history_hidden_dim = history_hidden_dim
        self.terrain_hidden_dim = terrain_hidden_dim
        self.operator_heads = operator_heads
        self.evolution_blocks = evolution_blocks
        self.dropout = dropout

    def forward(self, x, ls, topography):
        last_ozone = x[:, -1:, 0:1]
        return last_ozone.expand(-1, self.horizon, -1, -1, -1).clone()


def build_model(config):
    in_channels = _positive_int(config, "in_channels")
    if in_channels != 5:
        raise ValueError("in_channels must be exactly 5 in fixed channel order.")
    window = _positive_int(config, "window")
    if window < 2:
        raise ValueError("window must be at least 2 for future Ls continuation.")
    horizon = _positive_int(config, "horizon")
    history_hidden_dim = _positive_int(config, "history_hidden_dim")
    terrain_hidden_dim = _positive_int(config, "terrain_hidden_dim")
    operator_heads = _positive_int(config, "operator_heads")
    evolution_blocks = _positive_int(config, "evolution_blocks")
    if history_hidden_dim % operator_heads != 0:
        raise ValueError("history_hidden_dim must be divisible by operator_heads.")
    return SeasonalTopographicEvolutionOperator(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        history_hidden_dim=history_hidden_dim,
        terrain_hidden_dim=terrain_hidden_dim,
        operator_heads=operator_heads,
        evolution_blocks=evolution_blocks,
        dropout=_dropout(config),
    )
```

- [ ] **Step 4: Run the contract tests and confirm green**

Run the command from Step 2.

Expected: all three tests pass.

- [ ] **Step 5: Commit the contract scaffold**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: scaffold seasonal topographic operator model"
```

## Task 2: Spherical Grid Topology And Convolution

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing topology tests**

Append these methods to the test class:

```python
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
```

- [ ] **Step 2: Run the topology tests and confirm missing classes**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_spherical_neighbor_wraps_longitude tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_spherical_neighbor_reflects_over_pole_with_half_turn tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_spherical_distances_contract_toward_poles tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_spherical_conv_preserves_shape_and_backpropagates -v
```

Expected: errors stating that `SphericalGrid` and `SphericalConv2d` are absent.

- [ ] **Step 3: Implement exact global-grid neighbor lookup**

Add before the top-level model class:

```python
class SphericalGrid(nn.Module):
    MARS_RADIUS_METERS = 3_389_500.0

    def neighbor(self, x, dy, dx):
        height, width = x.shape[-2:]
        if width % 2 != 0:
            raise ValueError("spherical longitude width must be even.")
        if abs(dy) >= height:
            raise ValueError("latitude offset magnitude must be less than height.")

        rows = torch.arange(height, device=x.device).reshape(height, 1) + dy
        crossed_north = rows < 0
        crossed_south = rows >= height
        reflected_rows = torch.where(crossed_north, -rows - 1, rows)
        reflected_rows = torch.where(
            crossed_south, 2 * height - reflected_rows - 1, reflected_rows
        )
        crossed = crossed_north | crossed_south
        columns = torch.arange(width, device=x.device).reshape(1, width) + dx
        columns = columns + crossed.to(columns.dtype) * (width // 2)
        columns = columns.remainder(width)
        row_index = reflected_rows.expand(height, width)
        column_index = columns.expand(height, width)
        return x[..., row_index, column_index]

    def latitudes(self, height, device, dtype=torch.float32):
        rows = torch.arange(height, device=device, dtype=dtype)
        return torch.pi / 2 - (rows + 0.5) * torch.pi / height

    def distance_map(self, height, width, dy, dx, device, dtype=torch.float32):
        latitudes = self.latitudes(height, device, dtype).reshape(1, 1, height, 1)
        delta_latitude = torch.pi / height
        delta_longitude = 2 * torch.pi / width
        north_south = abs(dy) * delta_latitude
        east_west = abs(dx) * delta_longitude * torch.cos(latitudes).abs()
        angular = torch.sqrt(north_south**2 + east_west**2)
        return angular.expand(1, 1, height, width)


class SphericalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.grid = SphericalGrid()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size * kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        radius = self.kernel_size // 2
        patches = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                patches.append(self.grid.neighbor(x, dy, dx))
        stacked = torch.stack(patches, dim=2)
        output = torch.einsum("bckhw,ock->bohw", stacked, self.weight)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output
```

- [ ] **Step 4: Run the topology tests and the contract tests**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: the topology and contract tests pass; no unexpected errors.

- [ ] **Step 5: Commit spherical topology**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: add spherical global-grid topology"
```

## Task 3: Deterministic Terrain Geometry And Edge Features

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing terrain fixtures**

Append:

```python
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
        yy, xx = torch.meshgrid(torch.arange(9.0), torch.arange(16.0), indexing="ij")
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
```

- [ ] **Step 2: Run terrain tests and verify missing implementations**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_flat_terrain_produces_finite_zero_shape_descriptors tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_eastward_ramp_has_signed_east_slope tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_bowl_has_center_curvature_distinct_from_flat tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_edge_builder_creates_twenty_four_relational_neighbors -v
```

Expected: errors for missing `TerrainGeometryBank` and `TerrainEdgeBuilder`.

- [ ] **Step 3: Implement the approved 12-channel geometry bank**

Add:

```python
class TerrainGeometryBank(nn.Module):
    ELEVATION_SCALE_METERS = 10_000.0
    EPSILON = 1e-6

    def __init__(self):
        super().__init__()
        self.grid = SphericalGrid()

    def _neighbors(self, x, radius):
        offsets = (
            (-radius, -radius), (-radius, 0), (-radius, radius),
            (0, -radius), (0, radius),
            (radius, -radius), (radius, 0), (radius, radius),
        )
        return torch.stack(
            [self.grid.neighbor(x, dy, dx) for dy, dx in offsets], dim=2
        )

    def _local_stats(self, x, radius):
        samples = torch.cat((x.unsqueeze(2), self._neighbors(x, radius)), dim=2)
        mean = samples.mean(dim=2)
        variance = (samples - mean.unsqueeze(2)).square().mean(dim=2)
        return mean, torch.sqrt(variance + self.EPSILON) - self.EPSILON**0.5

    def forward(self, topography):
        height, width = topography.shape[-2:]
        east = self.grid.neighbor(topography, 0, 1)
        west = self.grid.neighbor(topography, 0, -1)
        north = self.grid.neighbor(topography, -1, 0)
        south = self.grid.neighbor(topography, 1, 0)
        latitude = self.grid.latitudes(
            height, topography.device, topography.dtype
        ).reshape(1, 1, height, 1)
        dx = (
            self.grid.MARS_RADIUS_METERS
            * torch.cos(latitude).abs()
            * (2 * torch.pi / width)
        ).clamp_min(1.0)
        dy = self.grid.MARS_RADIUS_METERS * torch.pi / height
        grade_east = (east - west) / (2 * dx)
        grade_north = (north - south) / (2 * dy)
        slope_east = torch.atan(grade_east) / (torch.pi / 2)
        slope_north = torch.atan(grade_north) / (torch.pi / 2)
        slope_mag = torch.sqrt(slope_east.square() + slope_north.square()) / 2**0.5
        grade_mag = torch.sqrt(grade_east.square() + grade_north.square())
        aspect_east = grade_east / (grade_mag + self.EPSILON)
        aspect_north = grade_north / (grade_mag + self.EPSILON)
        z_norm = topography / self.ELEVATION_SCALE_METERS
        cardinal_mean = torch.stack((east, west, north, south), dim=2).mean(dim=2)
        curvature = z_norm - cardinal_mean / self.ELEVATION_SCALE_METERS
        mean_1, _ = self._local_stats(topography, 1)
        mean_2, roughness_2 = self._local_stats(topography, 2)
        mean_4, roughness_4 = self._local_stats(topography, 4)
        return torch.cat(
            (
                z_norm,
                slope_east,
                slope_north,
                slope_mag,
                aspect_east,
                aspect_north,
                curvature,
                (topography - mean_1) / self.ELEVATION_SCALE_METERS,
                (topography - mean_2) / self.ELEVATION_SCALE_METERS,
                (topography - mean_4) / self.ELEVATION_SCALE_METERS,
                roughness_2 / self.ELEVATION_SCALE_METERS,
                roughness_4 / self.ELEVATION_SCALE_METERS,
            ),
            dim=1,
        )
```

- [ ] **Step 4: Implement multiscale directed edge features**

Add:

```python
class TerrainEdgeBuilder(nn.Module):
    RADII = (1, 2, 4)
    DIRECTIONS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    GEOMETRY_CHANNELS = 12
    EXTRA_CHANNELS = 7

    def __init__(self):
        super().__init__()
        self.grid = SphericalGrid()
        self.offsets = tuple(
            (radius * dy, radius * dx, radius, dy, dx)
            for radius in self.RADII
            for dy, dx in self.DIRECTIONS
        )
        self.edge_dim = 3 * self.GEOMETRY_CHANNELS + self.EXTRA_CHANNELS

    def forward(self, geometry):
        batch, _, height, width = geometry.shape
        edges = []
        for dy, dx, radius, unit_dy, unit_dx in self.offsets:
            neighbor = self.grid.neighbor(geometry, dy, dx)
            direction_norm = (unit_dx**2 + unit_dy**2) ** 0.5
            east_direction = unit_dx / direction_norm
            north_direction = -unit_dy / direction_norm
            alignment_p = (
                geometry[:, 4:5] * east_direction
                + geometry[:, 5:6] * north_direction
            )
            alignment_q = (
                neighbor[:, 4:5] * east_direction
                + neighbor[:, 5:6] * north_direction
            )
            distance = self.grid.distance_map(
                height, width, dy, dx, geometry.device, geometry.dtype
            ).expand(batch, -1, -1, -1)
            constants = geometry.new_empty(batch, 3, height, width)
            constants[:, 0].fill_(east_direction)
            constants[:, 1].fill_(north_direction)
            constants[:, 2].fill_(radius / max(self.RADII))
            extras = torch.cat(
                (
                    neighbor[:, 0:1] - geometry[:, 0:1],
                    alignment_p,
                    alignment_q,
                    distance,
                    constants,
                ),
                dim=1,
            )
            edges.append(
                torch.cat((geometry, neighbor, neighbor - geometry, extras), dim=1)
            )
        return torch.stack(edges, dim=1)
```

- [ ] **Step 5: Run terrain tests and check all current tests**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all current tests pass. If the bowl curvature sign differs because of the documented sign convention, keep `curvature = center - cardinal_mean` and correct only the fixture assertion to `assertLess` as shown above.

- [ ] **Step 6: Commit terrain geometry**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: derive spherical terrain geometry"
```

## Task 4: Future Ls And Bilinear Seasonal-Terrain Weights

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing Ls continuation and interaction tests**

Append:

```python
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
            edge_dim=43, terrain_hidden_dim=8, heads=4
        )
        edges = torch.randn(2, 24, 43, 8, 16)
        season = torch.randn(2, 8)
        encoded_edges = module.encode_edges(edges)
        weights = module(encoded_edges, season)
        self.assertEqual(weights.shape, (2, 24, 4, 8, 16))
        torch.testing.assert_close(
            weights.sum(dim=1), torch.ones(2, 4, 8, 16), atol=1e-6, rtol=1e-6
        )
        (weights.square().mean() + encoded_edges.square().mean()).backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_bilinear_terrain_response_changes_with_season(self):
        module = self.module.SeasonalTerrainWeights(
            edge_dim=43, terrain_hidden_dim=8, heads=2
        )
        edges = torch.randn(1, 24, 43, 4, 8)
        encoded_edges = module.encode_edges(edges)
        season_a = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        season_b = torch.tensor([[1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0]])
        weights_a = module(encoded_edges, season_a)
        weights_b = module(encoded_edges, season_b)
        self.assertGreater((weights_a - weights_b).abs().max().item(), 1e-7)
```

- [ ] **Step 2: Run the new tests and confirm missing classes**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_future_ls_continues_across_degree_wrap tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_future_ls_harmonics_have_four_orders tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_bilinear_weights_normalize_over_neighbors_and_backpropagate tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_bilinear_terrain_response_changes_with_season -v
```

Expected: errors for missing `FutureLsEncoder` and `SeasonalTerrainWeights`.

- [ ] **Step 3: Implement deterministic phase continuation and harmonic encoding**

Add:

```python
class FutureLsEncoder(nn.Module):
    def future_angles(self, ls, horizon):
        theta = torch.deg2rad(ls)
        differences = theta[:, 1:] - theta[:, :-1]
        wrapped = torch.atan2(torch.sin(differences), torch.cos(differences))
        recent = wrapped[:, -min(4, wrapped.shape[1]) :]
        angular_step = torch.atan2(
            torch.sin(recent).mean(dim=1), torch.cos(recent).mean(dim=1)
        )
        leads = torch.arange(
            1, horizon + 1, device=ls.device, dtype=ls.dtype
        ).reshape(1, horizon)
        return torch.remainder(theta[:, -1:] + angular_step[:, None] * leads, 2 * torch.pi)

    def harmonics(self, angles):
        values = []
        for order in range(1, 5):
            values.extend((torch.sin(order * angles), torch.cos(order * angles)))
        return torch.stack(values, dim=-1)

    def forward(self, ls, horizon):
        return self.harmonics(self.future_angles(ls, horizon))
```

- [ ] **Step 4: Implement low-rank bilinear edge-season weights**

Add:

```python
class SeasonalTerrainWeights(nn.Module):
    NEIGHBOR_COUNT = 24

    def __init__(self, edge_dim, terrain_hidden_dim, heads):
        super().__init__()
        self.heads = heads
        self.edge_projection = nn.Linear(edge_dim, terrain_hidden_dim)
        self.season_projection = nn.Linear(8, terrain_hidden_dim)
        self.output_projection = nn.Linear(terrain_hidden_dim, heads, bias=False)
        self.direction_scale_bias = nn.Parameter(
            torch.zeros(1, self.NEIGHBOR_COUNT, heads, 1, 1)
        )

    def encode_edges(self, edges):
        encoded = self.edge_projection(edges.permute(0, 1, 3, 4, 2))
        return encoded.permute(0, 1, 4, 2, 3)

    def forward(self, encoded_edges, season_harmonics):
        season = self.season_projection(season_harmonics)
        joint = encoded_edges * season[:, None, :, None, None]
        logits = self.output_projection(joint.permute(0, 1, 3, 4, 2))
        logits = logits.permute(0, 1, 4, 2, 3) + self.direction_scale_bias
        return torch.softmax(logits, dim=1)
```

- [ ] **Step 5: Run the new tests and complete regression file**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all tests pass. The interaction test must show a nonzero seasonal change without manually modifying trained weights.

- [ ] **Step 6: Commit seasonal-terrain coupling**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: couple future Ls with terrain edges"
```

## Task 5: Fixed-Semantics Full-Resolution History Encoder

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing separation and history-state tests**

Append:

```python
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
            self.assertGreater(inputs.grad[:, :, channel].abs().sum().item(), 0.0)
```

- [ ] **Step 2: Run history tests and confirm missing encoder**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_history_encoder_preserves_full_resolution tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_history_encoder_has_separate_ozone_and_forcing_stems tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_history_encoder_all_input_channels_receive_gradients -v
```

Expected: errors for missing `HistoryEncoder`.

- [ ] **Step 3: Implement the spherical ConvLSTM history path**

Add:

```python
class SphericalConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = SphericalConv2d(in_channels + hidden_dim, 4 * hidden_dim, 3)

    def forward(self, x, hidden, cell):
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((x, hidden), dim=1)
        ).chunk(4, dim=1)
        cell = (
            torch.sigmoid(forget_gate) * cell
            + torch.sigmoid(input_gate) * torch.tanh(candidate)
        )
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class HistoryEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        stem_dim = hidden_dim // 2
        self.ozone_stem = SphericalConv2d(1, stem_dim, 3)
        self.forcing_stem = SphericalConv2d(4, stem_dim, 3)
        self.fusion = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * stem_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.cell = SphericalConvLSTMCell(hidden_dim, hidden_dim)

    def forward(self, x):
        batch, steps, _, height, width = x.shape
        hidden = x.new_zeros(batch, self.cell.hidden_dim, height, width)
        cell = x.new_zeros(batch, self.cell.hidden_dim, height, width)
        for step in range(steps):
            ozone = self.ozone_stem(x[:, step, 0:1])
            forcing = self.forcing_stem(x[:, step, 1:5])
            fused = self.fusion(torch.cat((ozone, forcing), dim=1))
            hidden, cell = self.cell(fused, hidden, cell)
        return hidden, cell
```

- [ ] **Step 4: Run history and complete focused tests**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all current tests pass with finite gradients for every input channel.

- [ ] **Step 5: Commit the history encoder**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: encode fixed-semantics global history"
```

## Task 6: STEO Difference Messages

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing STEO message tests**

Append:

```python
    def test_steo_returns_state_shape_and_normalized_weights(self):
        operator = self.module.STEO(
            hidden_dim=16, edge_dim=43, terrain_hidden_dim=8, heads=4
        )
        state = torch.randn(2, 16, 8, 16)
        edges = torch.randn(2, 24, 43, 8, 16)
        season = torch.randn(2, 8)
        encoded_edges = operator.encode_edges(edges)
        update, weights = operator(state, encoded_edges, season)
        self.assertEqual(update.shape, state.shape)
        self.assertEqual(weights.shape, (2, 24, 4, 8, 16))
        torch.testing.assert_close(
            weights.sum(dim=1), torch.ones(2, 4, 8, 16), atol=1e-6, rtol=1e-6
        )

    def test_steo_constant_state_has_exact_zero_difference_update(self):
        operator = self.module.STEO(
            hidden_dim=16, edge_dim=43, terrain_hidden_dim=8, heads=4
        ).eval()
        state = torch.full((2, 16, 8, 16), 3.25)
        edges = torch.randn(2, 24, 43, 8, 16)
        season = torch.randn(2, 8)
        encoded_edges = operator.encode_edges(edges)
        update, _ = operator(state, encoded_edges, season)
        torch.testing.assert_close(update, torch.zeros_like(update), atol=0.0, rtol=0.0)

    def test_steo_backpropagates_to_state_edges_and_all_parameters(self):
        operator = self.module.STEO(
            hidden_dim=16, edge_dim=43, terrain_hidden_dim=8, heads=4
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
```

- [ ] **Step 2: Run STEO tests and confirm missing class**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_steo_returns_state_shape_and_normalized_weights tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_steo_constant_state_has_exact_zero_difference_update tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_steo_backpropagates_to_state_edges_and_all_parameters -v
```

Expected: errors for missing `STEO`.

- [ ] **Step 3: Implement multiscale, multihead state-difference messages**

Add:

```python
class STEO(nn.Module):
    def __init__(self, hidden_dim, edge_dim, terrain_hidden_dim, heads):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.grid = SphericalGrid()
        self.offsets = TerrainEdgeBuilder().offsets
        self.message_projection = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True)
        self.weight_network = SeasonalTerrainWeights(
            edge_dim=edge_dim,
            terrain_hidden_dim=terrain_hidden_dim,
            heads=heads,
        )
        self.merge = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)

    def encode_edges(self, edge_features):
        return self.weight_network.encode_edges(edge_features)

    def forward(self, state, encoded_edges, season_harmonics):
        batch, _, height, width = state.shape
        projected = self.message_projection(state).reshape(
            batch, self.heads, self.head_dim, height, width
        )
        differences = []
        for dy, dx, _, _, _ in self.offsets:
            neighbor = self.grid.neighbor(projected, dy, dx)
            differences.append(neighbor - projected)
        differences = torch.stack(differences, dim=1)
        weights = self.weight_network(encoded_edges, season_harmonics)
        message = (differences * weights.unsqueeze(3)).sum(dim=1)
        message = message.reshape(batch, self.hidden_dim, height, width)
        return self.merge(message), weights
```

- [ ] **Step 4: Run STEO and all focused tests**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all tests pass, including exact zero output for constant state.

- [ ] **Step 5: Commit STEO messages**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: add seasonal topographic evolution messages"
```

## Task 7: Pointwise Recurrent Evolution And End-To-End Forecast

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing integration and diagnostic tests**

Append:

```python
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
```

- [ ] **Step 2: Run integration tests and confirm the minimal model is insufficient**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_primary_model_forecasts_twenty_full_resolution_frames tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_platform_dry_run_shape_and_finite_backward tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_diagnostics_expose_lead_block_neighbor_head_weights -v
```

Expected: the primary shape test passes through the temporary persistence path;
the backward test fails because the output has no trainable model path, and the
diagnostic test errors because `forward_with_diagnostics` is absent.

- [ ] **Step 3: Implement pointwise dynamics and one recurrent block**

Add:

```python
class PointwiseDynamics(nn.Module):
    def __init__(self, hidden_dim, terrain_hidden_dim, dropout):
        super().__init__()
        self.season_projection = nn.Linear(8, terrain_hidden_dim)
        self.layers = nn.Sequential(
            nn.Conv2d(2 * hidden_dim + terrain_hidden_dim, 2 * hidden_dim, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(2 * hidden_dim, hidden_dim, 1),
        )

    def forward(self, state, history_context, season_harmonics):
        season = self.season_projection(season_harmonics).unsqueeze(-1).unsqueeze(-1)
        season = season.expand(-1, -1, state.shape[-2], state.shape[-1])
        return self.layers(torch.cat((state, history_context, season), dim=1))


class STEOEvolutionBlock(nn.Module):
    def __init__(
        self, hidden_dim, edge_dim, terrain_hidden_dim, heads, dropout
    ):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.operator = STEO(hidden_dim, edge_dim, terrain_hidden_dim, heads)
        self.pointwise = PointwiseDynamics(hidden_dim, terrain_hidden_dim, dropout)
        self.step_projection = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, state, history_context, encoded_edges, season_harmonics):
        normalized = self.norm(state)
        spatial, weights = self.operator(normalized, encoded_edges, season_harmonics)
        local = self.pointwise(normalized, history_context, season_harmonics)
        return state + self.step_projection(spatial + local), weights
```

- [ ] **Step 4: Replace the top-level model shell with full integration**

Replace `SeasonalTopographicEvolutionOperator.__init__` and `forward`, and add `_forecast` and `forward_with_diagnostics`:

```python
class SeasonalTopographicEvolutionOperator(nn.Module):
    def __init__(
        self,
        in_channels,
        window,
        horizon,
        history_hidden_dim,
        terrain_hidden_dim,
        operator_heads,
        evolution_blocks,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.history_hidden_dim = history_hidden_dim
        self.terrain_bank = TerrainGeometryBank()
        self.edge_builder = TerrainEdgeBuilder()
        self.history_encoder = HistoryEncoder(history_hidden_dim)
        self.ls_encoder = FutureLsEncoder()
        self.blocks = nn.ModuleList(
            [
                STEOEvolutionBlock(
                    hidden_dim=history_hidden_dim,
                    edge_dim=self.edge_builder.edge_dim,
                    terrain_hidden_dim=terrain_hidden_dim,
                    heads=operator_heads,
                    dropout=dropout,
                )
                for _ in range(evolution_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.GroupNorm(1, history_hidden_dim),
            nn.Conv2d(history_hidden_dim, history_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(history_hidden_dim, 1, 1),
        )

    def _validate_inputs(self, x, ls, topography):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must have shape [batch, window, 5, height, width].")
        parameter = next(self.parameters())
        if x.device != parameter.device:
            raise ValueError("x and model parameter device must match.")
        if not torch.is_floating_point(x) or x.dtype != parameter.dtype:
            raise ValueError("x and model parameters must use the same floating dtype.")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain finite floating-point values.")
        batch, window, channels, height, width = x.shape
        if window != self.window:
            raise ValueError(f"x window must be {self.window}, but received {window}.")
        if channels != 5:
            raise ValueError(f"x channels must be 5, but received {channels}.")
        if height < 8 or width < 8 or width % 2 != 0:
            raise ValueError("grid height must be at least 8 and width must be even and at least 8.")
        if not isinstance(ls, torch.Tensor) or tuple(ls.shape) != (batch, window):
            raise ValueError(f"ls shape must be ({batch}, {window}).")
        if not torch.is_floating_point(ls):
            raise ValueError("ls must contain floating-point values.")
        if ls.device != x.device:
            raise ValueError("x and ls device must match.")
        if not torch.isfinite(ls).all():
            raise ValueError("ls must contain finite floating-point values.")
        expected_topography = (batch, 1, height, width)
        if not isinstance(topography, torch.Tensor) or tuple(topography.shape) != expected_topography:
            raise ValueError(f"topography shape must be {expected_topography}.")
        if topography.dtype != torch.float32:
            raise ValueError("topography must contain float32 values.")
        if topography.device != x.device:
            raise ValueError("x and topography device must match.")
        if not torch.isfinite(topography).all():
            raise ValueError("topography must contain finite float32 values.")
        if topography.dtype != parameter.dtype:
            raise ValueError("x, topography, and model parameter dtypes must match.")

    def _forecast(self, x, ls, topography, return_diagnostics):
        self._validate_inputs(x, ls, topography)
        state, history_context = self.history_encoder(x)
        geometry = self.terrain_bank(topography)
        edges = self.edge_builder(geometry)
        harmonics = self.ls_encoder(ls, self.horizon)
        harmonics = harmonics.to(state.dtype)
        encoded_edges = [block.operator.encode_edges(edges) for block in self.blocks]
        last_ozone = x[:, -1, 0:1]
        outputs = []
        diagnostic_weights = []
        for lead in range(self.horizon):
            block_weights = []
            for block, block_edges in zip(self.blocks, encoded_edges):
                state, weights = block(
                    state, history_context, block_edges, harmonics[:, lead]
                )
                block_weights.append(weights)
            outputs.append(last_ozone + self.output_head(state))
            if return_diagnostics:
                diagnostic_weights.append(torch.stack(block_weights, dim=1))
        output = torch.stack(outputs, dim=1)
        if not return_diagnostics:
            return output
        weights = torch.stack(diagnostic_weights, dim=1)
        return output, weights

    def forward(self, x, ls, topography):
        return self._forecast(x, ls, topography, return_diagnostics=False)

    def forward_with_diagnostics(self, x, ls, topography):
        return self._forecast(x, ls, topography, return_diagnostics=True)
```

- [ ] **Step 5: Run end-to-end tests and inspect runtime**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all focused tests pass. Record the primary test runtime in the implementation notes; do not weaken the primary shape test to hide excessive runtime.

- [ ] **Step 6: Commit recurrent forecast integration**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: integrate recurrent STEO forecast"
```

## Task 8: Runtime Validation And Structural Constraints

**Files:**
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add exhaustive invalid-input and configuration tests**

Append:

```python
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
            ("window", lambda: model(torch.randn(2, 2, 5, 8, 16), ls[:, :2], topography)),
            ("channels", lambda: model(torch.randn(2, 3, 4, 8, 16), ls, topography)),
            ("finite", lambda: model(torch.full_like(x, float("nan")), ls, topography)),
            ("grid", lambda: model(torch.randn(2, 3, 5, 7, 16), ls, torch.zeros(2, 1, 7, 16))),
            ("grid", lambda: model(torch.randn(2, 3, 5, 8, 15), ls, torch.zeros(2, 1, 8, 15))),
            ("ls", lambda: model(x, torch.zeros(2, 2), topography)),
            ("ls", lambda: model(x, torch.zeros(2, 3, dtype=torch.long), topography)),
            ("ls", lambda: model(x, torch.full((2, 3), float("inf")), topography)),
            ("topography", lambda: model(x, ls, torch.zeros(2, 8, 16))),
            ("float32", lambda: model(x, ls, topography.double())),
            ("topography", lambda: model(x, ls, torch.full_like(topography, float("nan")))),
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
```

- [ ] **Step 2: Add upload-safety and no-bypass structural tests**

Append:

```python
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
        banned = {"open", "eval", "exec", "compile", "__import__", "system", "popen", "Popen", "run"}
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
```

- [ ] **Step 3: Run validation tests and correct message mismatches only**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_rejects_invalid_runtime_inputs tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_rejects_invalid_configuration tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_uses_only_upload_safe_imports_and_calls tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_future_non_operator_modules_are_pointwise tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_model_has_no_trainable_per_pixel_parameters -v
```

Expected: all tests pass. If one fails, change the implementation only when behavior is wrong; otherwise make the test regex match the existing clear field-specific message.

- [ ] **Step 4: Add deterministic evaluation and parameter reporting test**

Append:

```python
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
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(parameter_count, 0)
        print(f"STEO dry-run trainable parameters: {parameter_count}")
```

- [ ] **Step 5: Run the complete focused test module**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all focused tests pass with no skips and the dry-run parameter count is printed.

- [ ] **Step 6: Commit validation and structural coverage**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "test: verify STEO contract and structure"
```

## Task 9: Reproducible Operator Ablation Modes

**Files:**
- Modify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Modify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`

- [ ] **Step 1: Add failing schema and ablation-behavior tests**

Add `"operator_mode": "bilinear"` to `model_config`, then append:

```python
    def test_schema_exposes_exact_operator_ablation_modes(self):
        parameter = self.module.MODEL_SPEC["parameters"]["operator_mode"]
        self.assertEqual(parameter["type"], "select")
        self.assertEqual(parameter["default"], "bilinear")
        self.assertEqual(
            parameter["options"],
            ["fixed", "terrain_only", "ls_only", "separable", "bilinear", "ordinary_conv"],
        )

    def test_all_operator_modes_complete_the_same_dry_run(self):
        x = torch.randn(1, 3, 5, 8, 16)
        ls = torch.tensor([[20.0, 21.0, 22.0]])
        topography = torch.randn(1, 1, 8, 16) * 1000.0
        for mode in ("fixed", "terrain_only", "ls_only", "separable", "bilinear", "ordinary_conv"):
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
        season_a = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        season_b = torch.tensor([[1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0]])

        fixed = self.module.SeasonalTerrainWeights(43, 8, 2, mode="fixed")
        fixed_a = fixed(fixed.encode_edges(edges_a), season_a)
        fixed_b = fixed(fixed.encode_edges(edges_b), season_b)
        torch.testing.assert_close(fixed_a, fixed_b)

        terrain = self.module.SeasonalTerrainWeights(43, 8, 2, mode="terrain_only")
        terrain_a = terrain(terrain.encode_edges(edges_a), season_a)
        terrain_b = terrain(terrain.encode_edges(edges_a), season_b)
        torch.testing.assert_close(terrain_a, terrain_b)
        self.assertGreater(
            (terrain_a - terrain(terrain.encode_edges(edges_b), season_a)).abs().max().item(),
            1e-7,
        )

        ls_only = self.module.SeasonalTerrainWeights(43, 8, 2, mode="ls_only")
        ls_edges_a = ls_only.encode_edges(edges_a)
        ls_edges_b = ls_only.encode_edges(edges_b)
        torch.testing.assert_close(
            ls_only(ls_edges_a, season_a), ls_only(ls_edges_b, season_a)
        )
        self.assertGreater(
            (ls_only(ls_edges_a, season_a) - ls_only(ls_edges_a, season_b)).abs().max().item(),
            1e-7,
        )

    def test_ordinary_ablation_uses_spatial_convolution_block(self):
        model = self.module.build_model(model_config(operator_mode="ordinary_conv"))
        self.assertIsInstance(model.blocks[0], self.module.OrdinaryEvolutionBlock)

    def test_nonseasonal_modes_zero_the_future_season_context(self):
        harmonics = torch.randn(2, 3, 8)
        for mode in ("fixed", "terrain_only"):
            model = self.module.build_model(model_config(operator_mode=mode))
            conditioned = model._condition_harmonics(harmonics)
            torch.testing.assert_close(conditioned, torch.zeros_like(harmonics))
        for mode in ("ls_only", "separable", "bilinear", "ordinary_conv"):
            model = self.module.build_model(model_config(operator_mode=mode))
            torch.testing.assert_close(model._condition_harmonics(harmonics), harmonics)
```

- [ ] **Step 2: Run the ablation tests and confirm the schema is incomplete**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_schema_exposes_exact_operator_ablation_modes tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_all_operator_modes_complete_the_same_dry_run tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_ablation_dependencies_match_their_names tests.test_convlstm_seasonal_topographic_evolution_operator.SeasonalTopographicEvolutionOperatorTests.test_ordinary_ablation_uses_spatial_convolution_block -v
```

Expected: failures for the missing schema parameter, constructor `mode`, and
`OrdinaryEvolutionBlock`.

- [ ] **Step 3: Make seasonal-terrain weighting mode-explicit**

Replace `SeasonalTerrainWeights` with:

```python
class SeasonalTerrainWeights(nn.Module):
    NEIGHBOR_COUNT = 24
    VALID_MODES = ("fixed", "terrain_only", "ls_only", "separable", "bilinear")

    def __init__(self, edge_dim, terrain_hidden_dim, heads, mode="bilinear"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unsupported STEO weighting mode: {mode}.")
        self.mode = mode
        self.heads = heads
        self.terrain_hidden_dim = terrain_hidden_dim
        self.edge_projection = (
            nn.Linear(edge_dim, terrain_hidden_dim)
            if mode in {"terrain_only", "separable", "bilinear"}
            else None
        )
        self.season_projection = (
            nn.Linear(8, terrain_hidden_dim)
            if mode in {"ls_only", "separable", "bilinear"}
            else None
        )
        self.output_projection = (
            nn.Linear(terrain_hidden_dim, heads, bias=False)
            if mode != "fixed"
            else None
        )
        self.neighbor_embedding = (
            nn.Parameter(torch.empty(1, self.NEIGHBOR_COUNT, terrain_hidden_dim, 1, 1))
            if mode in {"ls_only", "separable"}
            else None
        )
        if self.neighbor_embedding is not None:
            nn.init.normal_(self.neighbor_embedding, mean=0.0, std=0.02)
        self.direction_scale_bias = nn.Parameter(
            torch.zeros(1, self.NEIGHBOR_COUNT, heads, 1, 1)
        )

    def encode_edges(self, edges):
        if self.edge_projection is None:
            return torch.zeros_like(edges[:, :, :1])
        encoded = self.edge_projection(edges.permute(0, 1, 3, 4, 2))
        return encoded.permute(0, 1, 4, 2, 3)

    def forward(self, encoded_edges, season_harmonics):
        batch, _, _, height, width = encoded_edges.shape
        if self.mode == "fixed":
            logits = self.direction_scale_bias.expand(batch, -1, -1, height, width)
            return torch.softmax(logits, dim=1)

        if self.season_projection is not None:
            season = self.season_projection(season_harmonics)
            season = season[:, None, :, None, None]
        if self.mode == "terrain_only":
            joint = encoded_edges
        elif self.mode == "ls_only":
            joint = self.neighbor_embedding * season
            joint = joint.expand(batch, -1, -1, height, width)
        elif self.mode == "separable":
            joint = encoded_edges + self.neighbor_embedding * season
        else:
            joint = encoded_edges * season

        logits = self.output_projection(joint.permute(0, 1, 3, 4, 2))
        logits = logits.permute(0, 1, 4, 2, 3) + self.direction_scale_bias
        return torch.softmax(logits, dim=1)
```

Modify `STEO.__init__` and `STEOEvolutionBlock.__init__` to accept
`operator_mode="bilinear"`, preserving direct-test compatibility, and pass it
to `SeasonalTerrainWeights`:

```python
self.weight_network = SeasonalTerrainWeights(
    edge_dim=edge_dim,
    terrain_hidden_dim=terrain_hidden_dim,
    heads=heads,
    mode=operator_mode,
)
```

- [ ] **Step 4: Implement the ordinary spatial recurrent ablation**

Add:

```python
class OrdinaryEvolutionBlock(nn.Module):
    def __init__(self, hidden_dim, terrain_hidden_dim, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.spatial = nn.Sequential(
            SphericalConv2d(hidden_dim, hidden_dim, 3, bias=True),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.pointwise = PointwiseDynamics(hidden_dim, terrain_hidden_dim, dropout)
        self.step_projection = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, state, history_context, encoded_edges, season_harmonics):
        normalized = self.norm(state)
        spatial = self.spatial(normalized)
        local = self.pointwise(normalized, history_context, season_harmonics)
        return state + self.step_projection(spatial + local), None
```

Add the schema field:

```python
"operator_mode": {
    "type": "select",
    "default": "bilinear",
    "options": [
        "fixed",
        "terrain_only",
        "ls_only",
        "separable",
        "bilinear",
        "ordinary_conv",
    ],
},
```

In `build_model`, validate and pass the selected value:

```python
valid_modes = {
    "fixed", "terrain_only", "ls_only", "separable", "bilinear", "ordinary_conv"
}
operator_mode = config["operator_mode"]
if operator_mode not in valid_modes:
    raise ValueError(f"operator_mode must be one of {sorted(valid_modes)}.")
```

Add `operator_mode` as the final argument of
`SeasonalTopographicEvolutionOperator.__init__`, then include it in the builder
call exactly as follows:

```python
return SeasonalTopographicEvolutionOperator(
    in_channels=in_channels,
    window=window,
    horizon=horizon,
    history_hidden_dim=history_hidden_dim,
    terrain_hidden_dim=terrain_hidden_dim,
    operator_heads=operator_heads,
    evolution_blocks=evolution_blocks,
    dropout=_dropout(config),
    operator_mode=operator_mode,
)
```

In the top-level constructor, store `self.operator_mode` and build blocks with:

```python
if operator_mode == "ordinary_conv":
    self.blocks = nn.ModuleList(
        [
            OrdinaryEvolutionBlock(
                history_hidden_dim, terrain_hidden_dim, dropout
            )
            for _ in range(evolution_blocks)
        ]
    )
else:
    self.blocks = nn.ModuleList(
        [
            STEOEvolutionBlock(
                hidden_dim=history_hidden_dim,
                edge_dim=self.edge_builder.edge_dim,
                terrain_hidden_dim=terrain_hidden_dim,
                heads=operator_heads,
                dropout=dropout,
                operator_mode=operator_mode,
            )
            for _ in range(evolution_blocks)
        ]
    )
```

Update `_forecast` so edge encodings are precomputed only for STEO blocks:

```python
if self.operator_mode == "ordinary_conv":
    encoded_edges = [None for _ in self.blocks]
else:
    encoded_edges = [block.operator.encode_edges(edges) for block in self.blocks]
```

Add a mode-specific seasonal-context method:

```python
def _condition_harmonics(self, harmonics):
    if self.operator_mode in {"fixed", "terrain_only"}:
        return torch.zeros_like(harmonics)
    return harmonics
```

Call it immediately after deterministic Ls encoding in `_forecast`:

```python
harmonics = self._condition_harmonics(harmonics.to(state.dtype))
```

This makes `fixed` independent of both Ls and terrain, makes `terrain_only`
independent of Ls across both the operator and pointwise branch, and leaves
`ls_only` independent of terrain through its mode-specific edge encoder.

When `return_diagnostics=True` and a block returns `None` weights, raise:

```python
raise ValueError("operator diagnostics are unavailable for ordinary_conv mode.")
```

- [ ] **Step 5: Run ablation and complete focused tests**

Run:

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all focused tests pass. The default bilinear model continues to pass
the no-spatial-bypass structural test; that test must not be run against the
explicit `ordinary_conv` control.

- [ ] **Step 6: Commit reproducible ablation modes**

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "feat: add STEO operator ablations"
```

## Task 10: Full Verification And Experimental Handoff

**Files:**
- Verify: `models/convlstm_seasonal_topographic_evolution_operator.py`
- Verify: `tests/test_convlstm_seasonal_topographic_evolution_operator.py`
- Reference: `docs/superpowers/specs/2026-08-26-seasonal-topographic-evolution-operator-design.md`

- [ ] **Step 1: Run the focused STEO suite from a fresh process**

```powershell
python -m unittest tests.test_convlstm_seasonal_topographic_evolution_operator -v
```

Expected: all STEO tests pass with zero failures, errors, and skips.

- [ ] **Step 2: Run the complete repository test suite**

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all repository tests pass. Investigate any regression before proceeding; do not dismiss failures as unrelated without reproducing them on the pre-STEO commit.

- [ ] **Step 3: Compile and inspect the upload artifact**

```powershell
python -m py_compile models/convlstm_seasonal_topographic_evolution_operator.py
git diff --check
git status --short
```

Expected: compilation exits successfully, `git diff --check` prints nothing, and status contains no uncommitted STEO model or test changes. Existing unrelated worktree entries may remain and must not be modified.

- [ ] **Step 4: Compare implementation against the approved specification**

Check each requirement explicitly:

```text
[ ] fixed five-channel semantics
[ ] full-resolution spherical history state
[ ] longitude periodicity and polar half-turn reflection
[ ] deterministic 12-channel terrain geometry
[ ] 24 multiscale directed edges
[ ] deterministic future Ls and four harmonic orders
[ ] bilinear non-separable season-terrain weights
[ ] per-head neighbor softmax
[ ] state-difference STEO messages
[ ] no ordinary future spatial bypass
[ ] recurrent shared-lead evolution
[ ] last-observation residual outputs
[ ] diagnostic operator weights
[ ] clear dtype, shape, device, and finite-value errors
[ ] upload-safe single-file implementation
```

Expected: every item maps to a passing test or a directly inspected model section.

- [ ] **Step 5: Record the training matrix for the experiment owner**

Use the exact approved sequence when scheduling platform runs:

```text
B0  Persistence
B1  ConvLSTM-SimVP
B2  Ls+MOLA joint gate with strength 1
B3  Fixed spherical neighbor operator
B4  Terrain-only operator
B5  Ls-only operator
B6  Separable additive Ls plus terrain operator
B7  Full bilinear STEO
B8  Parameter-matched ordinary spatial recurrent model
```

Run at least three matched seeds. Report overall and lead-wise RMSE first, then
Ls, elevation, slope, curvature, and latitude strata. Counterfactual terrain
runs are sensitivity diagnostics and must not be reported as causal accuracy.

- [ ] **Step 6: Make a final implementation commit only if verification produced edits**

When verification required a real code or test correction:

```powershell
git add -- models/convlstm_seasonal_topographic_evolution_operator.py tests/test_convlstm_seasonal_topographic_evolution_operator.py
git commit -m "fix: complete STEO verification"
```

When verification produced no edits, do not create an empty commit.
