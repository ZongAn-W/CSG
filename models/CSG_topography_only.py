"""Standalone terrain-only gating variant of original CSG.

G = tanh(W_g(GELU(f_topo(MOLA)))); E_out = E * (1 + s * G).
No dynamic values or Ls enter the gate generator. Historical input x still
feeds the prediction backbone; select channels on the training platform.
Gate strength is trainable, initialized to 1, with original STE [0, 1] clamp.
"""
import torch
from torch import nn


MODEL_SPEC = {'name': 'CSG_TopographyOnlyGate',
 'description': 'Terrain-only gate; no dynamic or seasonal gate inputs; strength '
                'initialized to 1.',
 'parameters': {'convlstm_hidden_dim': {'type': 'int',
                                        'default': 16,
                                        'min': 4,
                                        'max': 128},
                'spatial_hidden_dim': {'type': 'int',
                                       'default': 64,
                                       'min': 4,
                                       'max': 256},
                'temporal_hidden_dim': {'type': 'int',
                                        'default': 128,
                                        'min': 8,
                                        'max': 512},
                'num_temporal_blocks': {'type': 'int',
                                        'default': 2,
                                        'min': 1,
                                        'max': 8},
                'dropout': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 0.9},
                'gate_hidden_dim': {'type': 'int', 'default': 32, 'min': 4, 'max': 256},
                'initial_gate_strength': {'type': 'float',
                                          'default': 1.0,
                                          'min': 0.0,
                                          'max': 1.0}},
 'auxiliary_inputs': {'topography': {'required': True,
                                     'shape': ['batch', 1, 'height', 'width'],
                                     'dtype': 'float32',
                                     'unit': 'meter'}}}


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


class TopographyOnlyGate(nn.Module):
    """Spatial/channel gate shared across history, independent of dynamic values."""

    def __init__(self, channels, hidden_dim, initial_gate_strength):
        super().__init__()
        self.channels = channels
        self.topography_encoder = TopographyEncoder(hidden_dim)
        self.output_projection = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.gate_strength = nn.Parameter(torch.tensor(float(initial_gate_strength)))
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def current_strength(self):
        bounded = self.gate_strength.clamp(0.0, 1.0)
        return self.gate_strength + (bounded - self.gate_strength).detach()

    def gate_values(self, topography, output_size):
        terrain = self.topography_encoder(topography, output_size)
        # Retain the original joint-fusion GELU, with the dynamic term removed.
        return torch.tanh(self.output_projection(torch.nn.functional.gelu(terrain)))

    def forward(self, encoded, topography):
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
            raise ValueError("encoded must be rank-5 [batch, window, channels, height, width].")
        if encoded.shape[2] != self.channels:
            raise ValueError(f"encoded channels must be {self.channels}.")
        if topography.shape[0] != encoded.shape[0]:
            raise ValueError("encoded and topography batch sizes must match.")
        # Singleton time dimension broadcasts the same terrain gate to every step.
        gate = self.gate_values(topography, encoded.shape[-2:]).to(dtype=encoded.dtype).unsqueeze(1)
        scales = 1.0 + self.current_strength() * gate
        return scales * encoded, gate, scales


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
    def __init__(self, channels, dropout):
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

    def forward(self, x):
        normalized = self.norm(x)
        mixed = torch.cat(
            (self.branch_3x3(normalized), self.branch_5x5(normalized)), dim=1
        )
        return x + self.dropout(self.activation(self.merge(mixed)))


class TemporalTranslator(nn.Module):
    def __init__(
        self,
        window,
        horizon,
        spatial_hidden_dim,
        temporal_hidden_dim,
        num_blocks,
        dropout,
    ):
        super().__init__()
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.input_projection = nn.Conv2d(
            window * spatial_hidden_dim, temporal_hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            *[
                TemporalInceptionBlock(temporal_hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Conv2d(
            temporal_hidden_dim, horizon * spatial_hidden_dim, kernel_size=1
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.blocks(x)
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


class ConvLSTMTopographyOnlyGatedSimVP(nn.Module):
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
        gate_hidden_dim,
        initial_gate_strength,
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
        self.joint_gate = TopographyOnlyGate(
            channels=spatial_hidden_dim,
            hidden_dim=gate_hidden_dim,
            initial_gate_strength=initial_gate_strength,
        )
        self.temporal_translator = TemporalTranslator(
            window,
            horizon,
            spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_blocks,
            dropout,
        )
        self.spatial_decoder = SpatialDecoder(spatial_hidden_dim)

    def _validate_inputs(self, x, topography):
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

    def forward(self, x, topography):
        self._validate_inputs(x, topography)
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
        gated, _, _ = self.joint_gate(encoded, topography)
        translated = self.temporal_translator(
            gated.reshape(
                batch,
                window * self.spatial_hidden_dim,
                encoded_height,
                encoded_width,
            )
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


def _unit_interval(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number in the range [0, 1].")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be a number in the range [0, 1].")
    return float(value)


def build_model(config):
    in_channels = _positive_int(config, "in_channels")
    window = _positive_int(config, "window")
    horizon = _positive_int(config, "horizon")
    convlstm_hidden_dim = _positive_int(config, "convlstm_hidden_dim")
    spatial_hidden_dim = _positive_int(config, "spatial_hidden_dim")
    temporal_hidden_dim = _positive_int(config, "temporal_hidden_dim")
    num_temporal_blocks = _positive_int(config, "num_temporal_blocks")
    gate_hidden_dim = _positive_int(config, "gate_hidden_dim")
    initial_gate_strength = _unit_interval(config, "initial_gate_strength")
    dropout = config["dropout"]

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")

    return ConvLSTMTopographyOnlyGatedSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        convlstm_hidden_dim=convlstm_hidden_dim,
        spatial_hidden_dim=spatial_hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=float(dropout),
        gate_hidden_dim=gate_hidden_dim,
        initial_gate_strength=initial_gate_strength,
    )
