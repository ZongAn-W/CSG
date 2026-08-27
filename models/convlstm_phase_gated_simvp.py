import torch
from torch import nn


MODEL_SPEC = {
    "name": "ConvLSTMPhaseGatedSimVP",
    "description": (
        "ConvLSTM-SimVP with solar-longitude-conditioned memory and "
        "translation gates."
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
        "phase_context_dim": {
            "type": "int",
            "default": 32,
            "min": 4,
            "max": 256,
        },
        "initial_history_weight": {
            "type": "float",
            "default": 0.7,
            "min": 0.01,
            "max": 0.99,
        },
        "initial_translation_weight": {
            "type": "float",
            "default": 0.7,
            "min": 0.01,
            "max": 0.99,
        },
    },
}


def _initial_logit(probability):
    if not 0.0 < probability < 1.0:
        raise ValueError("initial gate weight must be between 0 and 1.")
    return torch.logit(torch.tensor(float(probability))).item()


class PhaseContextEncoder(nn.Module):
    def __init__(self, window, context_dim):
        super().__init__()
        self.window = window
        self.context_dim = context_dim
        self.layers = nn.Sequential(
            nn.Linear(window * 4, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

    def build_harmonic_features(self, ls):
        if ls.ndim != 2:
            raise ValueError("ls must have shape [batch, window].")
        if ls.shape[1] != self.window:
            raise ValueError(
                f"ls window must be {self.window}, but received {ls.shape[1]}."
            )

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

    def forward(self, ls):
        harmonic_features = self.build_harmonic_features(ls)
        flattened = harmonic_features.reshape(ls.shape[0], self.window * 4)
        return self.layers(flattened.to(dtype=self.layers[0].weight.dtype))


class PhaseConditionedMemoryGate(nn.Module):
    def __init__(self, hidden_dim, context_dim, initial_history_weight):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.phase_to_bias = nn.Linear(context_dim, hidden_dim)
        self.reset_parameters(initial_history_weight)

    def reset_parameters(self, initial_history_weight):
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.xavier_uniform_(self.phase_to_bias.weight, gain=0.1)
        nn.init.zeros_(self.phase_to_bias.bias)
        with torch.no_grad():
            self.gate.bias.fill_(_initial_logit(initial_history_weight))

    def forward(self, encoded, phase_context):
        if encoded.ndim != 5:
            raise ValueError(
                "encoded must have shape [batch, window, channels, height, width]."
            )
        batch, window, channels, height, width = encoded.shape
        if channels != self.hidden_dim:
            raise ValueError(
                f"encoded channels must be {self.hidden_dim}, but received {channels}."
            )

        feature_logits = self.gate(
            encoded.reshape(batch * window, channels, height, width)
        ).reshape(batch, window, channels, height, width)
        phase_bias = self.phase_to_bias(phase_context).reshape(
            batch, 1, channels, 1, 1
        )
        gate = torch.sigmoid(feature_logits + phase_bias)
        last_state = encoded[:, -1:].expand_as(encoded)
        admitted = gate * encoded + (1.0 - gate) * last_state
        return admitted, gate


class PhaseConditionedResidualGate(nn.Module):
    def __init__(self, hidden_dim, context_dim, initial_translation_weight):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.phase_to_bias = nn.Linear(context_dim, hidden_dim)
        self.reset_parameters(initial_translation_weight)

    def reset_parameters(self, initial_translation_weight):
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.xavier_uniform_(self.phase_to_bias.weight, gain=0.1)
        nn.init.zeros_(self.phase_to_bias.bias)
        with torch.no_grad():
            self.gate.bias.fill_(_initial_logit(initial_translation_weight))

    def forward(self, future_state, last_state, phase_context):
        if future_state.ndim != 5:
            raise ValueError(
                "future_state must have shape "
                "[batch, horizon, channels, height, width]."
            )
        if last_state.ndim != 4:
            raise ValueError(
                "last_state must have shape [batch, channels, height, width]."
            )
        batch, horizon, channels, height, width = future_state.shape
        if channels != self.hidden_dim:
            raise ValueError(
                f"future_state channels must be {self.hidden_dim}, "
                f"but received {channels}."
            )

        last_state_future = last_state.unsqueeze(1).expand_as(future_state)
        gate_input = torch.cat((future_state, last_state_future), dim=2)
        feature_logits = self.gate(
            gate_input.reshape(batch * horizon, 2 * channels, height, width)
        ).reshape(batch, horizon, channels, height, width)
        phase_bias = self.phase_to_bias(phase_context).reshape(
            batch, 1, channels, 1, 1
        )
        gate = torch.sigmoid(feature_logits + phase_bias)
        final_state = gate * future_state + (1.0 - gate) * last_state_future
        return final_state, gate


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


class ConvLSTMPhaseGatedSimVP(nn.Module):
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
        phase_context_dim,
        initial_history_weight,
        initial_translation_weight,
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
        self.phase_context_encoder = PhaseContextEncoder(
            window, phase_context_dim
        )
        self.memory_gate = PhaseConditionedMemoryGate(
            spatial_hidden_dim,
            phase_context_dim,
            initial_history_weight,
        )
        self.temporal_translator = TemporalTranslator(
            window,
            horizon,
            spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_blocks,
            dropout,
        )
        self.residual_gate = PhaseConditionedResidualGate(
            spatial_hidden_dim,
            phase_context_dim,
            initial_translation_weight,
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

        phase_context = self.phase_context_encoder(ls)
        admitted, _ = self.memory_gate(encoded, phase_context)
        translated = self.temporal_translator(
            admitted.reshape(
                batch,
                window * self.spatial_hidden_dim,
                encoded_height,
                encoded_width,
            )
        )
        future_state = translated.reshape(
            batch,
            self.horizon,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        final_state, _ = self.residual_gate(
            future_state, encoded[:, -1], phase_context
        )
        decoded = self.spatial_decoder(
            final_state.reshape(
                batch * self.horizon,
                self.spatial_hidden_dim,
                encoded_height,
                encoded_width,
            ),
            (height, width),
        )
        return decoded.reshape(batch, self.horizon, 1, height, width)


def _positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _probability(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number strictly between 0 and 1.")
    if not 0.0 < value < 1.0:
        raise ValueError(f"{key} must be a number strictly between 0 and 1.")
    return float(value)


def build_model(config):
    in_channels = _positive_int(config, "in_channels")
    window = _positive_int(config, "window")
    horizon = _positive_int(config, "horizon")
    convlstm_hidden_dim = _positive_int(config, "convlstm_hidden_dim")
    spatial_hidden_dim = _positive_int(config, "spatial_hidden_dim")
    temporal_hidden_dim = _positive_int(config, "temporal_hidden_dim")
    num_temporal_blocks = _positive_int(config, "num_temporal_blocks")
    phase_context_dim = _positive_int(config, "phase_context_dim")
    dropout = config["dropout"]

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")

    return ConvLSTMPhaseGatedSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        convlstm_hidden_dim=convlstm_hidden_dim,
        spatial_hidden_dim=spatial_hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=float(dropout),
        phase_context_dim=phase_context_dim,
        initial_history_weight=_probability(config, "initial_history_weight"),
        initial_translation_weight=_probability(
            config, "initial_translation_weight"
        ),
    )
