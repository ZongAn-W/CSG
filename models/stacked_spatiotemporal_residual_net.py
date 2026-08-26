import torch
from torch import nn


MODEL_SPEC = {
    "name": "StackedSpatiotemporalResidualNet",
    "description": (
        "Stacked two-layer ConvLSTM and full multiscale spatial residual blocks."
    ),
    "parameters": {
        "hidden_dim": {
            "type": "int",
            "default": 32,
            "min": 4,
            "max": 128,
        },
        "spatial_dim": {
            "type": "int",
            "default": 128,
            "min": 8,
            "max": 512,
        },
        "num_blocks": {
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


class StackedConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        if num_layers != 2:
            raise ValueError("StackedConvLSTM requires exactly two layers.")
        self.layers = nn.ModuleList(
            [
                ConvLSTMCell(input_dim, hidden_dim),
                ConvLSTMCell(hidden_dim, hidden_dim),
            ]
        )

    def forward(self, x):
        batch, steps, _, height, width = x.shape
        hidden_states = [
            x.new_zeros(batch, cell.hidden_dim, height, width)
            for cell in self.layers
        ]
        cell_states = [
            x.new_zeros(batch, cell.hidden_dim, height, width)
            for cell in self.layers
        ]
        outputs = []

        for step in range(steps):
            current = x[:, step]
            for index, layer in enumerate(self.layers):
                hidden_states[index], cell_states[index] = layer(
                    current, hidden_states[index], cell_states[index]
                )
                current = hidden_states[index]
            outputs.append(current)

        return torch.stack(outputs, dim=1)


class SpatiotemporalResidualBlock(nn.Module):
    def __init__(self, window, hidden_dim, spatial_dim, dropout):
        super().__init__()
        self.window = window
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        flattened_channels = window * hidden_dim

        self.temporal = StackedConvLSTM(hidden_dim, hidden_dim, num_layers=2)
        self.input_projection = nn.Conv2d(
            flattened_channels, spatial_dim, kernel_size=1
        )
        self.branch_3x3 = nn.Conv2d(
            spatial_dim, spatial_dim, kernel_size=3, padding=1
        )
        self.branch_5x5 = nn.Conv2d(
            spatial_dim, spatial_dim, kernel_size=5, padding=2
        )
        self.branch_7x7 = nn.Conv2d(
            spatial_dim, spatial_dim, kernel_size=7, padding=3
        )
        self.fusion = nn.Conv2d(3 * spatial_dim, spatial_dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, spatial_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(
            spatial_dim, flattened_channels, kernel_size=1
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                "Expected block input shape [batch, window, channels, height, width]."
            )
        batch, window, channels, height, width = x.shape
        if window != self.window or channels != self.hidden_dim:
            raise ValueError(
                "Block input window and channels do not match its configuration."
            )

        temporal = x + self.temporal(x)
        flat = temporal.reshape(batch, window * channels, height, width)
        base = self.input_projection(flat)
        multiscale = torch.cat(
            (
                self.branch_3x3(base),
                self.branch_5x5(base),
                self.branch_7x7(base),
            ),
            dim=1,
        )
        fused = self.dropout(self.norm(self.fusion(multiscale)))
        spatial = self.activation(base + fused)
        output = flat + self.output_projection(spatial)
        return output.reshape(batch, window, channels, height, width)


class StackedSpatiotemporalResidualNet(nn.Module):
    def __init__(
        self,
        in_channels,
        window,
        horizon,
        hidden_dim,
        spatial_dim,
        num_blocks,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window = window
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=3, padding=1
        )
        self.input_activation = nn.GELU()
        self.blocks = nn.Sequential(
            *[
                SpatiotemporalResidualBlock(
                    window=window,
                    hidden_dim=hidden_dim,
                    spatial_dim=spatial_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.time_projection = nn.Conv2d(
            window * hidden_dim,
            horizon * hidden_dim,
            kernel_size=1,
        )
        self.output_projection = nn.Conv2d(
            hidden_dim, 1, kernel_size=3, padding=1
        )

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

        encoded = self.input_projection(
            x.reshape(batch * window, channels, height, width)
        )
        encoded = self.input_activation(encoded).reshape(
            batch, window, self.hidden_dim, height, width
        )
        encoded = self.blocks(encoded)
        forecast = self.time_projection(
            encoded.reshape(batch, window * self.hidden_dim, height, width)
        )
        forecast = forecast.reshape(
            batch * self.horizon, self.hidden_dim, height, width
        )
        prediction = self.output_projection(forecast)
        return prediction.reshape(batch, self.horizon, 1, height, width)


def _positive_int(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def build_model(config):
    in_channels = _positive_int(config, "in_channels")
    window = _positive_int(config, "window")
    horizon = _positive_int(config, "horizon")
    hidden_dim = _positive_int(config, "hidden_dim")
    spatial_dim = _positive_int(config, "spatial_dim")
    num_blocks = _positive_int(config, "num_blocks")
    dropout = config["dropout"]

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be a number in the range [0, 1).")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a number in the range [0, 1).")

    return StackedSpatiotemporalResidualNet(
        in_channels=in_channels,
        window=window,
        horizon=horizon,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        num_blocks=num_blocks,
        dropout=float(dropout),
    )
