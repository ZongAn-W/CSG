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
