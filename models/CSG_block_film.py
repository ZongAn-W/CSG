"""Upload-ready per-block seasonal/terrain FiLM model (PyTorch only).

Each translator block: Z_next = Z + Dropout(GELU((1 + delta_gamma) * U + beta)).
U is the block's normalized multiscale convolution output. Shared conditions
use historical Ls only; independent block heads produce spatial affine maps.
FiLM heads use Xavier gain 0.1 for small, nonzero initial modulation.
"""

import torch
from torch import nn


MODEL_SPEC = {
    "name": "ConvLSTMLsTopographyBlockFiLMSimVP",
    "description": "ConvLSTM-SimVP with spatial seasonal-terrain FiLM inside every temporal residual block.",
    "parameters": {
        "convlstm_hidden_dim": {
            "type": "int",
            "default": 16,
            "min": 4,
            "max": 128,
        },
        "spatial_hidden_dim": {
            "type": "int",
            "default": 32,
            "min": 4,
            "max": 256,
        },
        "temporal_hidden_dim": {
            "type": "int",
            "default": 64,
            "min": 8,
            "max": 512,
        },
        "num_temporal_blocks": {
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
        "condition_hidden_dim": {
            "type": "int",
            "default": 32,
            "min": 4,
            "max": 256,
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


class LsHarmonicEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def build_harmonic_features(self, ls):
        if not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must be a rank-2 tensor [batch, window].")
        if not torch.is_floating_point(ls):
            raise TypeError("ls must use a floating-point dtype.")

        radians = torch.deg2rad(ls)
        return torch.stack(
            (
                torch.sin(radians),
                torch.cos(radians),
                torch.sin(2 * radians),
                torch.cos(2 * radians),
            ),
            dim=-1,
        )

    def forward(self, ls):
        harmonics = self.build_harmonic_features(ls)
        harmonics = harmonics.to(dtype=self.layers[0].weight.dtype)
        return self.layers(harmonics)


class TopographyEncoder(nn.Module):
    ELEVATION_SCALE_METERS = 10000.0

    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def scale_elevation(self, topography):
        return topography / self.ELEVATION_SCALE_METERS

    def forward(self, topography, output_size):
        topography = topography.to(dtype=self.layers[0].weight.dtype)
        encoded = self.layers(self.scale_elevation(topography))
        if encoded.shape[-2:] != output_size:
            encoded = torch.nn.functional.interpolate(
                encoded, size=output_size, mode="bilinear", align_corners=False
            )
        return encoded


class SeasonalTerrainCondition(nn.Module):
    """Encode the ordered historical Ls window and spatial terrain once."""

    def __init__(self, window, hidden_dim):
        super().__init__()
        self.ls_encoder = LsHarmonicEncoder(hidden_dim)
        self.window_projection = nn.Linear(window * hidden_dim, hidden_dim)
        self.topography_encoder = TopographyEncoder(hidden_dim)

    def forward(self, ls, topography, output_size):
        # Flattening preserves history order; the translator has no time axis.
        season = self.window_projection(self.ls_encoder(ls).flatten(1))
        terrain = self.topography_encoder(topography, output_size)
        return torch.nn.functional.gelu(
            terrain + season.to(dtype=terrain.dtype)[:, :, None, None]
        )


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=3,
            padding=1,
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


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(in_channels, hidden_dim)

    def forward(self, x):
        batch, steps, _, height, width = x.shape
        hidden = x.new_zeros(batch, self.hidden_dim, height, width)
        cell = x.new_zeros(batch, self.hidden_dim, height, width)
        outputs = []

        for step in range(steps):
            hidden, cell = self.cell(x[:, step], hidden, cell)
            outputs.append(hidden)

        return torch.stack(outputs, dim=1)


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class TemporalInceptionBlock(nn.Module):
    def __init__(self, channels, dropout, condition_hidden_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.branch_3x3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.branch_5x5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
        )
        self.merge = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        # Each block owns its affine head; no tanh or strength clipping.
        self.film_projection = nn.Conv2d(condition_hidden_dim, 2 * channels, 1)
        nn.init.xavier_uniform_(self.film_projection.weight, gain=0.1)
        nn.init.zeros_(self.film_projection.bias)

    def forward(self, x, condition):
        normalized = self.norm(x)
        mixed = torch.cat(
            (self.branch_3x3(normalized), self.branch_5x5(normalized)), dim=1
        )
        delta_gamma, beta = self.film_projection(condition).chunk(2, dim=1)
        modulated = (1.0 + delta_gamma) * self.merge(mixed) + beta
        return x + self.dropout(self.activation(modulated))


class TemporalTranslator(nn.Module):
    def __init__(
        self,
        window,
        horizon,
        spatial_hidden_dim,
        temporal_hidden_dim,
        num_blocks,
        dropout,
        condition_hidden_dim,
    ):
        super().__init__()
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.input_projection = nn.Conv2d(
            window * spatial_hidden_dim, temporal_hidden_dim, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            [
                TemporalInceptionBlock(temporal_hidden_dim, dropout, condition_hidden_dim)
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Conv2d(
            temporal_hidden_dim, horizon * spatial_hidden_dim, kernel_size=1
        )

    def forward(self, x, condition):
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x, condition)
        return self.output_projection(x)


class SpatialDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, output_size):
        x = self.layers(x)
        if x.shape[-2:] != output_size:
            x = torch.nn.functional.interpolate(
                x, size=output_size, mode="bilinear", align_corners=False
            )
        return x


class ConvLSTMLsTopographyBlockFiLMSimVP(nn.Module):
    def __init__(
        self,
        in_channels,
        window,
        horizon,
        convlstm_hidden_dim,
        spatial_hidden_dim,
        temporal_hidden_dim,
        num_temporal_blocks,
        dropout,
        condition_hidden_dim,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.convlstm = ConvLSTMEncoder(in_channels, convlstm_hidden_dim)
        self.spatial_encoder = SpatialEncoder(
            convlstm_hidden_dim, spatial_hidden_dim
        )
        self.condition_encoder = SeasonalTerrainCondition(window, condition_hidden_dim)
        self.temporal_translator = TemporalTranslator(
            window,
            horizon,
            spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_blocks,
            dropout,
            condition_hidden_dim,
        )
        self.spatial_decoder = SpatialDecoder(spatial_hidden_dim)

    def _validate_inputs(self, x, ls, topography):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError(
                "x must have shape [batch, window, channels, height, width]."
            )

        batch, window, channels, height, width = x.shape
        if window != self.window:
            raise ValueError(
                f"x window must be {self.window}, but received {window}."
            )
        if channels != self.in_channels:
            raise ValueError(
                f"x channels must be {self.in_channels}, but received {channels}."
            )
        if ls is None or not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must be a tensor with shape [batch, window].")
        if tuple(ls.shape) != (batch, self.window):
            raise ValueError(
                f"ls shape must be ({batch}, {self.window}), "
                f"but received {tuple(ls.shape)}."
            )
        if not torch.is_floating_point(ls):
            raise ValueError("ls dtype must be floating point.")
        if ls.device != x.device:
            raise ValueError(
                f"x and ls device must match, but received {x.device} and {ls.device}."
            )
        if not torch.isfinite(ls).all():
            raise ValueError("ls values must all be finite; NaN and Inf are invalid.")
        if (
            topography is None
            or not isinstance(topography, torch.Tensor)
            or topography.ndim != 4
        ):
            raise ValueError(
                "topography must be a tensor with shape "
                "[batch, 1, height, width]."
            )
        expected_topography_shape = (batch, 1, height, width)
        if tuple(topography.shape) != expected_topography_shape:
            raise ValueError(
                f"topography shape must be {expected_topography_shape}, "
                f"but received {tuple(topography.shape)}."
            )
        if not torch.is_floating_point(topography):
            raise ValueError("topography dtype must be floating point.")
        if topography.device != x.device:
            raise ValueError(
                "x and topography device must match, but received "
                f"{x.device} and {topography.device}."
            )
        if not torch.isfinite(topography).all():
            raise ValueError(
                "topography values must all be finite; NaN and Inf are invalid."
            )

    def forward(self, x, ls, topography):
        self._validate_inputs(x, ls, topography)
        batch, window, _, height, width = x.shape

        recurrent_features = self.convlstm(x)
        encoded = self.spatial_encoder(
            recurrent_features.reshape(
                batch * window,
                recurrent_features.shape[2],
                height,
                width,
            )
        )
        encoded_height, encoded_width = encoded.shape[-2:]
        encoded = encoded.reshape(
            batch,
            window,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        condition = self.condition_encoder(ls, topography, (encoded_height, encoded_width))
        translated = self.temporal_translator(
            encoded.reshape(
                batch,
                window * self.spatial_hidden_dim,
                encoded_height,
                encoded_width,
            ),
            condition,
        )
        translated = translated.reshape(
            batch * self.horizon,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        decoded = self.spatial_decoder(translated, (height, width))
        return decoded.reshape(batch, self.horizon, 1, height, width)


def _positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def build_model(config):
    in_channels = _positive_int(config, "in_channels")
    window = _positive_int(config, "window")
    horizon = _positive_int(config, "horizon")
    convlstm_hidden_dim = _positive_int(config, "convlstm_hidden_dim")
    spatial_hidden_dim = _positive_int(config, "spatial_hidden_dim")
    temporal_hidden_dim = _positive_int(config, "temporal_hidden_dim")
    num_temporal_blocks = _positive_int(config, "num_temporal_blocks")
    condition_hidden_dim = _positive_int(config, "condition_hidden_dim")
    dropout = config["dropout"]

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")

    return ConvLSTMLsTopographyBlockFiLMSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        convlstm_hidden_dim=convlstm_hidden_dim,
        spatial_hidden_dim=spatial_hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=float(dropout),
        condition_hidden_dim=condition_hidden_dim,
    )
