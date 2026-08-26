import torch
from torch import nn


MODEL_SPEC = {
    "name": "SeasonalTopographicEvolutionOperator",
    "description": (
        "Full-resolution recurrent Mars ozone model with a bilinear "
        "solar-longitude and terrain-conditioned spherical evolution operator."
    ),
    "parameters": {
        "history_hidden_dim": {
            "type": "int",
            "default": 32,
            "min": 8,
            "max": 128,
        },
        "terrain_hidden_dim": {
            "type": "int",
            "default": 24,
            "min": 8,
            "max": 128,
        },
        "operator_heads": {
            "type": "int",
            "default": 4,
            "min": 1,
            "max": 16,
        },
        "evolution_blocks": {
            "type": "int",
            "default": 2,
            "min": 1,
            "max": 4,
        },
        "dropout": {
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 0.9,
        },
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
            crossed_south,
            2 * height - reflected_rows - 1,
            reflected_rows,
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

    def distance_map(
        self,
        height,
        width,
        dy,
        dx,
        device,
        dtype=torch.float32,
    ):
        latitudes = self.latitudes(height, device, dtype).reshape(
            1, 1, height, 1
        )
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
        self.in_channels = in_channels
        self.out_channels = out_channels
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


class TerrainGeometryBank(nn.Module):
    ELEVATION_SCALE_METERS = 10_000.0
    EPSILON = 1e-6

    def __init__(self):
        super().__init__()
        self.grid = SphericalGrid()

    def _neighbors(self, x, radius):
        offsets = (
            (-radius, -radius),
            (-radius, 0),
            (-radius, radius),
            (0, -radius),
            (0, radius),
            (radius, -radius),
            (radius, 0),
            (radius, radius),
        )
        return torch.stack(
            [self.grid.neighbor(x, dy, dx) for dy, dx in offsets],
            dim=2,
        )

    def _local_stats(self, x, radius):
        samples = torch.cat((x.unsqueeze(2), self._neighbors(x, radius)), dim=2)
        mean = samples.mean(dim=2)
        variance = (samples - mean.unsqueeze(2)).square().mean(dim=2)
        roughness = torch.sqrt(variance + self.EPSILON) - self.EPSILON**0.5
        return mean, roughness

    def forward(self, topography):
        height, width = topography.shape[-2:]
        east = self.grid.neighbor(topography, 0, 1)
        west = self.grid.neighbor(topography, 0, -1)
        north = self.grid.neighbor(topography, -1, 0)
        south = self.grid.neighbor(topography, 1, 0)
        latitude = self.grid.latitudes(
            height,
            topography.device,
            topography.dtype,
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
        slope_mag = torch.sqrt(
            slope_east.square() + slope_north.square()
        ) / 2**0.5
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


class TerrainEdgeBuilder(nn.Module):
    RADII = (1, 2, 4)
    DIRECTIONS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
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
                height,
                width,
                dy,
                dx,
                geometry.device,
                geometry.dtype,
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
                torch.cat(
                    (geometry, neighbor, neighbor - geometry, extras),
                    dim=1,
                )
            )
        return torch.stack(edges, dim=1)


class FutureLsEncoder(nn.Module):
    def future_angles(self, ls, horizon):
        theta = torch.deg2rad(ls)
        differences = theta[:, 1:] - theta[:, :-1]
        wrapped = torch.atan2(torch.sin(differences), torch.cos(differences))
        recent = wrapped[:, -min(4, wrapped.shape[1]) :]
        angular_step = torch.atan2(
            torch.sin(recent).mean(dim=1),
            torch.cos(recent).mean(dim=1),
        )
        leads = torch.arange(
            1,
            horizon + 1,
            device=ls.device,
            dtype=ls.dtype,
        ).reshape(1, horizon)
        return torch.remainder(
            theta[:, -1:] + angular_step[:, None] * leads,
            2 * torch.pi,
        )

    def harmonics(self, angles):
        values = []
        for order in range(1, 5):
            values.extend(
                (
                    torch.sin(order * angles),
                    torch.cos(order * angles),
                )
            )
        return torch.stack(values, dim=-1)

    def forward(self, ls, horizon):
        return self.harmonics(self.future_angles(ls, horizon))


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


class SphericalConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = SphericalConv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            3,
        )

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
        self.message_projection = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            1,
            bias=True,
        )
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
            batch,
            self.heads,
            self.head_dim,
            height,
            width,
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


class PointwiseDynamics(nn.Module):
    def __init__(self, hidden_dim, terrain_hidden_dim, dropout):
        super().__init__()
        self.season_projection = nn.Linear(8, terrain_hidden_dim)
        self.layers = nn.Sequential(
            nn.Conv2d(
                2 * hidden_dim + terrain_hidden_dim,
                2 * hidden_dim,
                1,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(2 * hidden_dim, hidden_dim, 1),
        )

    def forward(self, state, history_context, season_harmonics):
        season = self.season_projection(season_harmonics)
        season = season.unsqueeze(-1).unsqueeze(-1)
        season = season.expand(
            -1,
            -1,
            state.shape[-2],
            state.shape[-1],
        )
        return self.layers(
            torch.cat((state, history_context, season), dim=1)
        )


class STEOEvolutionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        edge_dim,
        terrain_hidden_dim,
        heads,
        dropout,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.operator = STEO(
            hidden_dim,
            edge_dim,
            terrain_hidden_dim,
            heads,
        )
        self.pointwise = PointwiseDynamics(
            hidden_dim,
            terrain_hidden_dim,
            dropout,
        )
        self.step_projection = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        state,
        history_context,
        encoded_edges,
        season_harmonics,
    ):
        normalized = self.norm(state)
        spatial, weights = self.operator(
            normalized,
            encoded_edges,
            season_harmonics,
        )
        local = self.pointwise(
            normalized,
            history_context,
            season_harmonics,
        )
        return state + self.step_projection(spatial + local), weights


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

    def _forecast(self, x, ls, topography, return_diagnostics):
        state, history_context = self.history_encoder(x)
        geometry = self.terrain_bank(topography)
        edges = self.edge_builder(geometry)
        harmonics = self.ls_encoder(ls, self.horizon).to(state.dtype)
        encoded_edges = [
            block.operator.encode_edges(edges) for block in self.blocks
        ]
        last_ozone = x[:, -1, 0:1]
        outputs = []
        diagnostic_weights = []
        for lead in range(self.horizon):
            block_weights = []
            for block, block_edges in zip(self.blocks, encoded_edges):
                state, weights = block(
                    state,
                    history_context,
                    block_edges,
                    harmonics[:, lead],
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
