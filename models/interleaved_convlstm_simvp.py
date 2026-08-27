import torch
from torch import nn


MODEL_SPEC = {
    "name": "InterleavedConvLSTMSimVP",
    "description": (
        "SimVP blocks with an independent ConvLSTM temporal update before "
        "each spatial multiscale residual block."
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


class InterleavedTemporalBlock(nn.Module):
    def __init__(self, channels, convlstm_hidden_dim, dropout):
        super().__init__()
        self.channels = channels
        self.convlstm = ConvLSTMEncoder(channels, convlstm_hidden_dim)
        self.state_projection = nn.Conv2d(
            convlstm_hidden_dim, channels, kernel_size=1
        )
        self.simvp_block = TemporalInceptionBlock(channels, dropout)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        recurrent = self.convlstm(x).reshape(
            batch * steps,
            self.convlstm.hidden_dim,
            height,
            width,
        )
        residual = self.state_projection(recurrent).reshape(
            batch,
            steps,
            channels,
            height,
            width,
        )
        features = (x + residual).reshape(batch * steps, channels, height, width)
        return self.simvp_block(features).reshape(
            batch,
            steps,
            channels,
            height,
            width,
        )


class TemporalTranslator(nn.Module):
    def __init__(self, channels, convlstm_hidden_dim, num_blocks, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                InterleavedTemporalBlock(
                    channels,
                    convlstm_hidden_dim,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


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
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return x


class InterleavedConvLSTMSimVP(nn.Module):
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.spatial_encoder = SpatialEncoder(in_channels, spatial_hidden_dim)
        self.input_projection = nn.Conv2d(
            spatial_hidden_dim,
            temporal_hidden_dim,
            kernel_size=1,
        )
        self.temporal_translator = TemporalTranslator(
            temporal_hidden_dim,
            convlstm_hidden_dim,
            num_temporal_blocks,
            dropout,
        )
        self.output_projection = nn.Conv2d(
            window * temporal_hidden_dim,
            horizon * spatial_hidden_dim,
            kernel_size=1,
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

        encoded = self.spatial_encoder(
            x.reshape(batch * window, channels, height, width)
        )
        encoded_height, encoded_width = encoded.shape[-2:]
        encoded = self.input_projection(encoded).reshape(
            batch,
            window,
            self.temporal_hidden_dim,
            encoded_height,
            encoded_width,
        )
        translated = self.temporal_translator(encoded)
        forecast = self.output_projection(
            translated.reshape(
                batch,
                window * self.temporal_hidden_dim,
                encoded_height,
                encoded_width,
            )
        ).reshape(
            batch * self.horizon,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        decoded = self.spatial_decoder(forecast, (height, width))
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
    dropout = config["dropout"]

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")

    return InterleavedConvLSTMSimVP(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        convlstm_hidden_dim=convlstm_hidden_dim,
        spatial_hidden_dim=spatial_hidden_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=float(dropout),
    )
