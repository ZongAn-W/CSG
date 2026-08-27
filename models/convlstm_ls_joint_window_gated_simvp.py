import torch
from torch import nn


MODEL_SPEC = {
    "name": "ConvLSTMLsJointWindowGatedSimVP",
    "description": (
        "ConvLSTM-SimVP with jointly computed Ls-conditioned residual "
        "scaling for each input window."
    ),
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        }
    },
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
        "window_gate_hidden_dim": {
            "type": "int",
            "default": 32,
            "min": 4,
            "max": 256,
        },
        "initial_window_strength": {
            "type": "float",
            "default": 0.05,
            "min": 0.0,
            "max": 1.0,
        },
    },
}


class LsConditionedJointWindowGate(nn.Module):
    def __init__(
        self,
        window,
        channels,
        hidden_dim,
        initial_window_strength,
    ):
        super().__init__()
        self.window = window
        self.channels = channels
        self.scorer = nn.Sequential(
            nn.Linear(window * (channels + 4), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, window),
        )
        self.window_strength = nn.Parameter(
            torch.tensor(float(initial_window_strength))
        )
        nn.init.xavier_uniform_(self.scorer[-1].weight, gain=0.01)
        nn.init.zeros_(self.scorer[-1].bias)

    def build_harmonic_features(self, ls):
        if not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must have shape [batch, window].")
        if ls.shape[1] != self.window:
            raise ValueError(
                f"ls window must be {self.window}, but received {ls.shape[1]}."
            )
        if not torch.is_floating_point(ls):
            raise ValueError("ls dtype must be floating point.")

        ls_rad = ls * (torch.pi / 180.0)
        return torch.stack(
            (
                torch.sin(ls_rad),
                torch.cos(ls_rad),
                torch.sin(2.0 * ls_rad),
                torch.cos(2.0 * ls_rad),
            ),
            dim=-1,
        )

    def current_strength(self):
        bounded = self.window_strength.clamp(0.0, 1.0)
        # Keep the forward value bounded without freezing an out-of-range update.
        return self.window_strength + (bounded - self.window_strength).detach()

    def window_weights(self, encoded, ls):
        if encoded.ndim != 5:
            raise ValueError(
                "encoded must have shape [batch, window, channels, height, width]."
            )
        batch, window, channels, _, _ = encoded.shape
        if window != self.window:
            raise ValueError(
                f"encoded window must be {self.window}, but received {window}."
            )
        if channels != self.channels:
            raise ValueError(
                f"encoded channels must be {self.channels}, but received {channels}."
            )
        if tuple(ls.shape) != (batch, window):
            raise ValueError(
                f"ls shape must be ({batch}, {window}), but received {tuple(ls.shape)}."
            )

        descriptors = encoded.mean(dim=(-2, -1))
        harmonics = self.build_harmonic_features(ls).to(dtype=descriptors.dtype)
        joint_input = torch.cat((descriptors, harmonics), dim=-1)
        logits = self.scorer(joint_input.reshape(batch, window * (channels + 4)))
        return torch.sigmoid(logits)

    def forward(self, encoded, ls):
        weights = self.window_weights(encoded, ls)
        strength = self.current_strength()
        scales = 1.0 + strength * (2.0 * weights - 1.0)
        weighted = scales.reshape(encoded.shape[0], self.window, 1, 1, 1) * encoded
        return weighted, weights, scales


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


class ConvLSTMLsJointWindowGatedSimVP(nn.Module):
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
        window_gate_hidden_dim,
        initial_window_strength,
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
        self.window_gate = LsConditionedJointWindowGate(
            window,
            spatial_hidden_dim,
            window_gate_hidden_dim,
            initial_window_strength,
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

    def _validate_inputs(self, x, ls):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError(
                "x must have shape [batch, window, channels, height, width]."
            )

        batch, window, channels, _, _ = x.shape
        if window != self.window:
            raise ValueError(
                f"x window must be {self.window}, but received {window}."
            )
        if channels != self.in_channels:
            raise ValueError(
                f"x channels must be {self.in_channels}, but received {channels}."
            )
        if ls is None:
            raise ValueError("ls is required and must have shape [batch, window].")
        if not isinstance(ls, torch.Tensor) or ls.ndim != 2:
            raise ValueError("ls must be a tensor with shape [batch, window].")
        if tuple(ls.shape) != (batch, self.window):
            raise ValueError(
                f"ls shape must be ({batch}, {self.window}), "
                f"but received {tuple(ls.shape)}."
            )
        if not torch.is_floating_point(ls):
            raise ValueError("ls dtype must be floating point.")
        if x.device != ls.device:
            raise ValueError(
                f"x and ls device must match, but received {x.device} and {ls.device}."
            )
        if not torch.isfinite(ls).all():
            raise ValueError("ls values must all be finite; NaN and Inf are invalid.")

    def forward(self, x, ls=None):
        self._validate_inputs(x, ls)
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
        weighted, _, _ = self.window_gate(encoded, ls)
        translated = self.temporal_translator(
            weighted.reshape(
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
    window_gate_hidden_dim = _positive_int(config, "window_gate_hidden_dim")
    dropout = config["dropout"]

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")

    return ConvLSTMLsJointWindowGatedSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        convlstm_hidden_dim=convlstm_hidden_dim,
        spatial_hidden_dim=spatial_hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=float(dropout),
        window_gate_hidden_dim=window_gate_hidden_dim,
        initial_window_strength=_unit_interval(
            config, "initial_window_strength"
        ),
    )
