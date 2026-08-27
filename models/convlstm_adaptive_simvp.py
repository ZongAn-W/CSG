import torch
from torch import nn


MODEL_SPEC = {
    "name": "ConvLSTMAdaptiveSimVP",
    "description": (
        "ConvLSTM-SimVP with horizon-conditioned channel-wise adaptive "
        "temporal fusion."
    ),
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
        "attention_initial_strength": {
            "type": "float",
            "default": 0.2,
            "min": 0.01,
            "max": 0.99,
        },
    },
}


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


class ChannelTemporalFusion(nn.Module):
    def __init__(self, window, horizon, channels, initial_strength):
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.channels = channels
        self.descriptor_norm = nn.LayerNorm(channels)
        self.key_projection = nn.Linear(channels, channels)
        self.score_projection = nn.Linear(channels, channels, bias=False)
        self.forecast_query = nn.Parameter(torch.empty(horizon, channels))
        self.history_bias = nn.Parameter(torch.zeros(window, channels))
        self.mix_logit = nn.Parameter(
            torch.full((channels,), initial_strength).logit()
        )
        nn.init.normal_(self.forecast_query, mean=0.0, std=1.0)

    def attention_weights(self, encoded):
        if encoded.ndim != 5:
            raise ValueError(
                "Expected encoded shape [batch, window, channels, height, width]."
            )
        if encoded.shape[1] != self.window or encoded.shape[2] != self.channels:
            raise ValueError("Encoded window or channel count does not match fusion.")

        descriptors = self.descriptor_norm(encoded.mean(dim=(-2, -1)))
        keys = self.key_projection(descriptors)
        joint = (
            keys.unsqueeze(1)
            * self.forecast_query.unsqueeze(0).unsqueeze(2)
            + self.history_bias.unsqueeze(0).unsqueeze(0)
        )
        logits = 2.0 * self.score_projection(torch.tanh(joint))
        adaptive = torch.softmax(logits, dim=2)

        strength = torch.sigmoid(self.mix_logit).view(1, 1, 1, self.channels)
        uniform = 1.0 / self.window
        return (1.0 - strength) * uniform + strength * adaptive

    def forward(self, encoded):
        weights = self.attention_weights(encoded)
        return torch.einsum("bkts,btshw->bkshw", weights, encoded)


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


class TemporalRefiner(nn.Module):
    def __init__(
        self,
        horizon,
        spatial_hidden_dim,
        temporal_hidden_dim,
        num_blocks,
        dropout,
    ):
        super().__init__()
        forecast_channels = horizon * spatial_hidden_dim
        self.input_projection = nn.Conv2d(
            forecast_channels, temporal_hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            *[
                TemporalInceptionBlock(temporal_hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Conv2d(
            temporal_hidden_dim, forecast_channels, kernel_size=1
        )

    def forward(self, x):
        correction = self.input_projection(x)
        correction = self.blocks(correction)
        correction = self.output_projection(correction)
        return x + correction


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


class ConvLSTMAdaptiveSimVP(nn.Module):
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
        attention_initial_strength,
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
        self.temporal_fusion = ChannelTemporalFusion(
            window,
            horizon,
            spatial_hidden_dim,
            attention_initial_strength,
        )
        self.temporal_refiner = TemporalRefiner(
            horizon,
            spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_blocks,
            dropout,
        )
        self.spatial_decoder = SpatialDecoder(spatial_hidden_dim)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                "Expected input shape [batch, window, channels, height, width]."
            )

        batch, window, channels, height, width = x.shape
        if window != self.window:
            raise ValueError(
                f"Expected window={self.window}, but received window={window}."
            )
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but received {channels}."
            )

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

        fused = self.temporal_fusion(encoded)
        fused = fused.reshape(
            batch,
            self.horizon * self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        refined = self.temporal_refiner(fused)
        refined = refined.reshape(
            batch * self.horizon,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        decoded = self.spatial_decoder(refined, (height, width))
        return decoded.reshape(batch, self.horizon, 1, height, width)


def _positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _bounded_float(config, key, lower, upper, lower_inclusive):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number in the required range.")
    if lower_inclusive:
        valid = lower <= value < upper
    else:
        valid = lower < value < upper
    if not valid:
        raise ValueError(f"{key} must be a number in the required range.")
    return float(value)


def build_model(config):
    in_channels = _positive_int(config, "in_channels")
    window = _positive_int(config, "window")
    if window < 2:
        raise ValueError("window must be at least 2 for adaptive temporal fusion.")
    horizon = _positive_int(config, "horizon")
    convlstm_hidden_dim = _positive_int(config, "convlstm_hidden_dim")
    spatial_hidden_dim = _positive_int(config, "spatial_hidden_dim")
    temporal_hidden_dim = _positive_int(config, "temporal_hidden_dim")
    num_temporal_blocks = _positive_int(config, "num_temporal_blocks")
    dropout = _bounded_float(config, "dropout", 0.0, 1.0, True)
    attention_initial_strength = _bounded_float(
        config,
        "attention_initial_strength",
        0.0,
        1.0,
        False,
    )

    return ConvLSTMAdaptiveSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        convlstm_hidden_dim=convlstm_hidden_dim,
        spatial_hidden_dim=spatial_hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=dropout,
        attention_initial_strength=attention_initial_strength,
    )
