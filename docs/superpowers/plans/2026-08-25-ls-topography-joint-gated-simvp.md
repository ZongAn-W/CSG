# Ls-Topography Joint-Gated ConvLSTM-SimVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an AresVision-compatible ConvLSTM-SimVP model whose encoded history is modulated by a small joint solar-longitude and MOLA topography residual gate.

**Architecture:** Copy the existing ConvLSTM-SimVP main path into an independent upload file. Encode per-step Ls harmonics and static signed MOLA elevation, fuse both with dynamic encoded features into a `[B,T,S,h,w]` gate, and multiply the history by a near-identity residual scale before the unchanged SimVP translator.

**Tech Stack:** Python, PyTorch, standard-library `unittest`, AST-based upload safety checks.

---

## File Structure

- Create `models/convlstm_ls_topography_joint_gated_simvp.py`: standalone upload-safe model, auxiliary encoders, joint gate, runtime validation, and builder.
- Create `tests/test_convlstm_ls_topography_joint_gated_simvp.py`: contract, unit, integration, regression, gradient, safety, and parameter-budget tests.
- Read but do not modify `models/convlstm_simvp.py`: source of the unchanged main-path classes.
- Read but do not modify `models/convlstm_ls_joint_window_gated_simvp.py`: source pattern for Ls validation, harmonic encoding, and bounded residual strength.

## Task 1: Create The Standalone Upload Contract

**Files:**
- Create: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`
- Create: `models/convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Write the failing contract test**

Create the test file with the loader, primary configuration, and exact metadata assertions:

```python
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "convlstm_ls_topography_joint_gated_simvp.py"
BASELINE_MODEL_PATH = ROOT / "models" / "convlstm_simvp.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_config(**overrides):
    config = {
        "in_channels": 5,
        "window": 20,
        "horizon": 20,
        "height": 8,
        "width": 16,
        "selected_channels": [0, 1, 2, 3, 4],
        "convlstm_hidden_dim": 16,
        "spatial_hidden_dim": 32,
        "temporal_hidden_dim": 64,
        "num_temporal_blocks": 3,
        "dropout": 0.1,
        "gate_hidden_dim": 32,
        "initial_gate_strength": 0.05,
    }
    config.update(overrides)
    return config


class LsTopographyJointGatedSimVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("joint_gate_model", MODEL_PATH) if MODEL_PATH.exists() else None
        cls.baseline = load_module("joint_gate_baseline", BASELINE_MODEL_PATH)

    def setUp(self):
        if self._testMethodName != "test_model_file_exists" and self.module is None:
            self.skipTest("model file does not exist yet")

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.is_file(), f"missing model file: {MODEL_PATH}")

    def test_exports_exact_upload_contract(self):
        expected_auxiliary_inputs = {
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
        }
        expected_parameters = {
            "convlstm_hidden_dim",
            "spatial_hidden_dim",
            "temporal_hidden_dim",
            "num_temporal_blocks",
            "dropout",
            "gate_hidden_dim",
            "initial_gate_strength",
        }

        self.assertEqual(self.module.MODEL_SPEC["auxiliary_inputs"], expected_auxiliary_inputs)
        self.assertEqual(set(self.module.MODEL_SPEC["parameters"]), expected_parameters)
        self.assertEqual(
            list(inspect.signature(self.module.ConvLSTMLsTopographyJointGatedSimVP.forward).parameters),
            ["self", "x", "ls", "topography"],
        )
        self.assertIsInstance(self.module.build_model(model_config()), nn.Module)
```

- [ ] **Step 2: Run the test to verify the red state**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_model_file_exists -v
```

Expected: `FAIL` with `missing model file`.

- [ ] **Step 3: Copy the exact baseline before adding the contract**

Run:

```powershell
Copy-Item -LiteralPath "models\convlstm_simvp.py" -Destination "models\convlstm_ls_topography_joint_gated_simvp.py"
```

In the copied file, replace `MODEL_SPEC` with the exact block below:

```python
MODEL_SPEC = {
    "name": "ConvLSTMLsTopographyJointGatedSimVP",
    "description": (
        "ConvLSTM-SimVP with a joint solar-longitude and MOLA "
        "topography residual gate."
    ),
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
    "parameters": {
        "convlstm_hidden_dim": {"type": "int", "default": 16, "min": 4, "max": 128},
        "spatial_hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
        "temporal_hidden_dim": {"type": "int", "default": 64, "min": 8, "max": 512},
        "num_temporal_blocks": {"type": "int", "default": 3, "min": 1, "max": 8},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
        "gate_hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 256},
        "initial_gate_strength": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0},
    },
}
```

Replace the copied top-level model class with this baseline-equivalent scaffold.
The two condition parameters are stored but do not alter the main path until
Task 4:

```python
class ConvLSTMLsTopographyJointGatedSimVP(nn.Module):
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
        self.gate_hidden_dim = gate_hidden_dim
        self.initial_gate_strength = initial_gate_strength
        self.convlstm = ConvLSTMEncoder(in_channels, convlstm_hidden_dim)
        self.spatial_encoder = SpatialEncoder(
            convlstm_hidden_dim, spatial_hidden_dim
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

    def forward(self, x, ls, topography):
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
            window * self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        translated = self.temporal_translator(encoded)
        translated = translated.reshape(
            batch * self.horizon,
            self.spatial_hidden_dim,
            encoded_height,
            encoded_width,
        )
        decoded = self.spatial_decoder(translated, (height, width))
        return decoded.reshape(batch, self.horizon, 1, height, width)
```

Add this helper beside `_positive_int`:

```python
def _unit_interval(config, key):
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number in the range [0, 1].")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be a number in the range [0, 1].")
    return float(value)
```

Replace the copied `build_model` with this complete builder:

```python
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

    return ConvLSTMLsTopographyJointGatedSimVP(
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
```

- [ ] **Step 4: Run the contract tests**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: both contract tests `PASS`.

- [ ] **Step 5: Commit the standalone contract**

```powershell
git add -- "models/convlstm_ls_topography_joint_gated_simvp.py" "tests/test_convlstm_ls_topography_joint_gated_simvp.py"
git commit -m "feat: scaffold LS topography gated model"
```

## Task 2: Implement The Ls And Topography Encoders

**Files:**
- Modify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`
- Modify: `models/convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Add failing encoder tests**

Add these methods to `LsTopographyJointGatedSimVPTests`:

```python
def test_ls_encoder_builds_exact_first_and_second_harmonics(self):
    encoder = self.module.LsHarmonicEncoder(hidden_dim=8)
    ls = torch.tensor([[0.0, 90.0, 180.0, 270.0]])
    actual = encoder.build_harmonic_features(ls)
    expected = torch.tensor(
        [[
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, -1.0],
        ]]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)

def test_topography_encoder_preserves_signed_scale_and_matches_encoded_size(self):
    encoder = self.module.TopographyEncoder(hidden_dim=8)
    topography = torch.tensor(
        [[[[-10000.0, 0.0, 20000.0], [-5000.0, 5000.0, 10000.0], [0.0, 0.0, 0.0]]]]
    )
    scaled = encoder.scale_elevation(topography)
    actual = encoder(topography, output_size=(2, 2))

    torch.testing.assert_close(
        scaled[0, 0, 0], torch.tensor([-1.0, 0.0, 2.0]), atol=0.0, rtol=0.0
    )
    self.assertEqual(tuple(actual.shape), (1, 8, 2, 2))
    self.assertTrue(torch.isfinite(actual).all())
```

- [ ] **Step 2: Run the encoder tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_ls_encoder_builds_exact_first_and_second_harmonics tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_topography_encoder_preserves_signed_scale_and_matches_encoded_size -v
```

Expected: `ERROR` because `LsHarmonicEncoder` and `TopographyEncoder` do not exist.

- [ ] **Step 3: Implement both condition encoders**

Insert these classes before `ConvLSTMCell`:

```python
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
            raise ValueError("ls must have shape [batch, window].")
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

    def forward(self, ls):
        harmonics = self.build_harmonic_features(ls)
        return self.layers(harmonics.to(dtype=self.layers[0].weight.dtype))


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
        encoded = self.layers(self.scale_elevation(topography))
        if encoded.shape[-2:] != output_size:
            encoded = torch.nn.functional.interpolate(
                encoded,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return encoded
```

- [ ] **Step 4: Run the encoder tests and full focused file**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: all current tests `PASS`.

- [ ] **Step 5: Commit the condition encoders**

```powershell
git add -- "models/convlstm_ls_topography_joint_gated_simvp.py" "tests/test_convlstm_ls_topography_joint_gated_simvp.py"
git commit -m "feat: encode LS and MOLA conditions"
```

## Task 3: Implement The Joint Spatiotemporal Residual Gate

**Files:**
- Modify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`
- Modify: `models/convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Add failing gate behavior and gradient tests**

Add these test methods:

```python
def test_joint_gate_has_expected_shape_range_and_condition_sensitivity(self):
    torch.manual_seed(7)
    gate = self.module.JointSpatiotemporalGate(
        spatial_hidden_dim=6,
        gate_hidden_dim=8,
        initial_gate_strength=0.05,
    )
    encoded = torch.randn(2, 3, 6, 4, 5, requires_grad=True)
    ls_a = torch.zeros(2, 3)
    ls_b = torch.tensor([[30.0, 60.0, 90.0], [120.0, 150.0, 180.0]])
    topography_a = torch.zeros(2, 1, 8, 10)
    topography_b = torch.linspace(-8000.0, 16000.0, 160).reshape(2, 1, 8, 10)

    gated, values_a, scales = gate(encoded, ls_a, topography_a)
    _, values_ls, _ = gate(encoded, ls_b, topography_a)
    _, values_topography, _ = gate(encoded, ls_a, topography_b)

    self.assertEqual(tuple(values_a.shape), (2, 3, 6, 4, 5))
    self.assertEqual(tuple(gated.shape), tuple(encoded.shape))
    self.assertGreaterEqual(float(scales.min().detach()), 0.95 - 1e-6)
    self.assertLessEqual(float(scales.max().detach()), 1.05 + 1e-6)
    self.assertGreater(float((values_a - values_ls).abs().max().detach()), 1e-7)
    self.assertGreater(float((values_a - values_topography).abs().max().detach()), 1e-7)

    gated.square().mean().backward()
    for name, parameter in gate.named_parameters():
        self.assertIsNotNone(parameter.grad, name)
        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

def test_joint_gate_is_nonseparable_across_ls_and_topography(self):
    torch.manual_seed(11)
    gate = self.module.JointSpatiotemporalGate(6, 8, 0.05).eval()
    encoded = torch.randn(1, 3, 6, 4, 5)
    ls_a = torch.zeros(1, 3)
    ls_b = torch.full((1, 3), 90.0)
    low = torch.full((1, 1, 8, 10), -5000.0)
    high = torch.full((1, 1, 8, 10), 15000.0)

    ls_effect_low = gate.gate_values(encoded, ls_b, low) - gate.gate_values(encoded, ls_a, low)
    ls_effect_high = gate.gate_values(encoded, ls_b, high) - gate.gate_values(encoded, ls_a, high)

    self.assertGreater(float((ls_effect_low - ls_effect_high).abs().max()), 1e-8)
```

- [ ] **Step 2: Run the gate tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_joint_gate_has_expected_shape_range_and_condition_sensitivity tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_joint_gate_is_nonseparable_across_ls_and_topography -v
```

Expected: `ERROR` because `JointSpatiotemporalGate` does not exist.

- [ ] **Step 3: Implement the joint gate**

Insert this class after `TopographyEncoder`:

```python
class JointSpatiotemporalGate(nn.Module):
    def __init__(
        self,
        spatial_hidden_dim,
        gate_hidden_dim,
        initial_gate_strength,
    ):
        super().__init__()
        self.spatial_hidden_dim = spatial_hidden_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.feature_projection = nn.Conv2d(
            spatial_hidden_dim, gate_hidden_dim, kernel_size=1
        )
        self.ls_encoder = LsHarmonicEncoder(gate_hidden_dim)
        self.topography_encoder = TopographyEncoder(gate_hidden_dim)
        self.output_projection = nn.Conv2d(
            gate_hidden_dim, spatial_hidden_dim, kernel_size=1
        )
        self.gate_strength = nn.Parameter(
            torch.tensor(float(initial_gate_strength))
        )
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def current_strength(self):
        bounded = self.gate_strength.clamp(0.0, 1.0)
        return self.gate_strength + (bounded - self.gate_strength).detach()

    def gate_values(self, encoded, ls, topography):
        if encoded.ndim != 5:
            raise ValueError(
                "encoded must have shape [batch, window, channels, height, width]."
            )
        batch, window, channels, height, width = encoded.shape
        if channels != self.spatial_hidden_dim:
            raise ValueError(
                f"encoded channels must be {self.spatial_hidden_dim}, "
                f"but received {channels}."
            )

        feature_context = self.feature_projection(
            encoded.reshape(batch * window, channels, height, width)
        ).reshape(batch, window, self.gate_hidden_dim, height, width)
        ls_context = self.ls_encoder(ls).reshape(
            batch, window, self.gate_hidden_dim, 1, 1
        )
        terrain_context = self.topography_encoder(
            topography, output_size=(height, width)
        ).unsqueeze(1)
        joint_context = torch.nn.functional.gelu(
            feature_context + ls_context + terrain_context
        )
        gate = self.output_projection(
            joint_context.reshape(
                batch * window, self.gate_hidden_dim, height, width
            )
        ).reshape(batch, window, channels, height, width)
        return torch.tanh(gate)

    def forward(self, encoded, ls, topography):
        gate = self.gate_values(encoded, ls, topography)
        scales = 1.0 + self.current_strength() * gate
        return scales * encoded, gate, scales
```

- [ ] **Step 4: Run the gate tests and focused suite**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: all current tests `PASS`, including finite gradients for every gate parameter.

- [ ] **Step 5: Commit the joint gate**

```powershell
git add -- "models/convlstm_ls_topography_joint_gated_simvp.py" "tests/test_convlstm_ls_topography_joint_gated_simvp.py"
git commit -m "feat: add joint spatiotemporal residual gate"
```

## Task 4: Integrate The Gate And Enforce Runtime Contracts

**Files:**
- Modify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`
- Modify: `models/convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Add failing end-to-end and baseline-fallback tests**

Add these methods:

```python
def test_matches_platform_dry_run_with_both_auxiliary_inputs(self):
    model = self.module.build_model(
        model_config(
            in_channels=1,
            selected_channels=[0],
            window=3,
            horizon=3,
            height=8,
            width=16,
        )
    ).eval()
    with torch.no_grad():
        actual = model(
            torch.randn(2, 3, 1, 8, 16),
            torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]]),
            torch.randn(2, 1, 8, 16) * 1000.0,
        )
    self.assertEqual(tuple(actual.shape), (2, 3, 1, 8, 16))

def test_forward_supports_primary_and_odd_multichannel_shapes(self):
    primary = self.module.build_model(model_config()).eval()
    x = torch.randn(1, 20, 5, 8, 16)
    ls = torch.linspace(0.0, 190.0, 20).unsqueeze(0)
    topography = torch.linspace(-8000.0, 16000.0, 128).reshape(1, 1, 8, 16)
    with torch.no_grad():
        actual = primary(x, ls, topography)
    self.assertEqual(tuple(actual.shape), (1, 20, 1, 8, 16))

    odd = self.module.build_model(
        model_config(
            in_channels=2,
            selected_channels=[0, 1],
            window=4,
            horizon=2,
            height=9,
            width=15,
            convlstm_hidden_dim=4,
            spatial_hidden_dim=6,
            temporal_hidden_dim=8,
            num_temporal_blocks=1,
            dropout=0.0,
            gate_hidden_dim=8,
        )
    )
    odd_output = odd(
        torch.randn(1, 4, 2, 9, 15),
        torch.tensor([[0.0, 30.0, 60.0, 90.0]]),
        torch.randn(1, 1, 9, 15) * 1000.0,
    )
    odd_output.square().mean().backward()
    self.assertEqual(tuple(odd_output.shape), (1, 2, 1, 9, 15))
    for name, parameter in odd.named_parameters():
        self.assertIsNotNone(parameter.grad, name)
        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

def test_zero_strength_exactly_matches_baseline_with_shared_weights(self):
    baseline_config = {
        key: value
        for key, value in model_config(window=3, horizon=2, height=8, width=16).items()
        if key not in {"gate_hidden_dim", "initial_gate_strength"}
    }
    baseline = self.baseline.build_model(baseline_config).eval()
    joint = self.module.build_model(
        model_config(window=3, horizon=2, height=8, width=16, initial_gate_strength=0.0)
    ).eval()
    joint_state = joint.state_dict()
    for name, value in baseline.state_dict().items():
        joint_state[name] = value
    joint.load_state_dict(joint_state)

    x = torch.randn(2, 3, 5, 8, 16)
    ls = torch.tensor([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]])
    topography = torch.randn(2, 1, 8, 16) * 2000.0
    with torch.no_grad():
        expected = baseline(x)
        actual = joint(x, ls, topography)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

def test_rejects_invalid_auxiliary_inputs(self):
    model = self.module.build_model(model_config(window=3, horizon=2))
    x = torch.randn(2, 3, 5, 8, 16)
    valid_ls = torch.zeros(2, 3)
    valid_topography = torch.zeros(2, 1, 8, 16)
    cases = [
        ("x", lambda: model(torch.randn(2, 3, 5, 8), valid_ls, valid_topography)),
        ("window", lambda: model(torch.randn(2, 2, 5, 8, 16), valid_ls[:, :2], valid_topography)),
        ("channels", lambda: model(torch.randn(2, 3, 4, 8, 16), valid_ls, valid_topography)),
        ("ls", lambda: model(x, None, valid_topography)),
        ("ls", lambda: model(x, torch.zeros(2, 2), valid_topography)),
        ("ls", lambda: model(x, torch.zeros(2, 3, dtype=torch.long), valid_topography)),
        ("ls", lambda: model(x, torch.tensor([[0.0, 1.0, float("nan")]]).expand(2, -1), valid_topography)),
        ("device", lambda: model(x, torch.zeros(2, 3, device="meta"), valid_topography)),
        ("topography", lambda: model(x, valid_ls, None)),
        ("topography", lambda: model(x, valid_ls, torch.zeros(2, 8, 16))),
        ("topography", lambda: model(x, valid_ls, torch.zeros(2, 2, 8, 16))),
        ("topography", lambda: model(x, valid_ls, torch.zeros(2, 1, 7, 16))),
        ("topography", lambda: model(x, valid_ls, torch.zeros(2, 1, 8, 16, dtype=torch.long))),
        ("topography", lambda: model(x, valid_ls, torch.full((2, 1, 8, 16), float("inf")))),
        ("device", lambda: model(x, valid_ls, torch.zeros(2, 1, 8, 16, device="meta"))),
    ]
    for label, call in cases:
        with self.subTest(label=label):
            with self.assertRaisesRegex(ValueError, label):
                call()

def test_rejects_invalid_gate_configuration(self):
    with self.assertRaisesRegex(ValueError, "positive integer"):
        self.module.build_model(model_config(num_temporal_blocks=0))
    with self.assertRaisesRegex(ValueError, "dropout"):
        self.module.build_model(model_config(dropout=1.0))
    with self.assertRaisesRegex(ValueError, "gate_hidden_dim"):
        self.module.build_model(model_config(gate_hidden_dim=0))
    with self.assertRaisesRegex(ValueError, "initial_gate_strength"):
        self.module.build_model(model_config(initial_gate_strength=-0.01))
    with self.assertRaisesRegex(ValueError, "initial_gate_strength"):
        self.module.build_model(model_config(initial_gate_strength=1.01))
```

- [ ] **Step 2: Run the integration tests to verify the red state**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: at least one `FAIL` because the main model does not yet validate or use the auxiliary inputs.

- [ ] **Step 3: Replace the scaffold constructor attributes with the gate**

In `ConvLSTMLsTopographyJointGatedSimVP.__init__`, keep every copied baseline
module unchanged and replace the scaffold gate fields with:

```python
self.joint_gate = JointSpatiotemporalGate(
    spatial_hidden_dim=spatial_hidden_dim,
    gate_hidden_dim=gate_hidden_dim,
    initial_gate_strength=initial_gate_strength,
)
```

- [ ] **Step 4: Add complete runtime validation**

Add this method to the top-level model:

```python
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
            f"ls shape must be ({batch}, {self.window}), but received {tuple(ls.shape)}."
        )
    if not torch.is_floating_point(ls):
        raise ValueError("ls dtype must be floating point.")
    if ls.device != x.device:
        raise ValueError(
            f"x and ls device must match, but received {x.device} and {ls.device}."
        )
    if not torch.isfinite(ls).all():
        raise ValueError("ls values must all be finite; NaN and Inf are invalid.")
    if topography is None or not isinstance(topography, torch.Tensor) or topography.ndim != 4:
        raise ValueError(
            "topography must be a tensor with shape [batch, 1, height, width]."
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
```

- [ ] **Step 5: Integrate the gate into the copied baseline forward path**

Replace the top-level `forward` method with the complete flow below. Keep the
copied encoder, translator, and decoder calls unchanged around the insertion:

```python
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
    gated, _, _ = self.joint_gate(encoded, ls, topography)
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
```

- [ ] **Step 6: Run the integration and focused tests**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: all current tests `PASS`, including exact zero-strength fallback.

- [ ] **Step 7: Commit the integrated model**

```powershell
git add -- "models/convlstm_ls_topography_joint_gated_simvp.py" "tests/test_convlstm_ls_topography_joint_gated_simvp.py"
git commit -m "feat: integrate LS topography gate into SimVP"
```

## Task 5: Lock Main-Path Equivalence, Safety, And Parameter Budget

**Files:**
- Modify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Add safety, AST-equivalence, and budget tests**

Add `ast` to the imports and these helpers near the loaders:

```python
import ast


def class_ast(module_path, class_name):
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"missing class {class_name} in {module_path}")


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
```

Add these test methods:

```python
def test_main_path_classes_are_ast_identical_to_baseline(self):
    class_names = (
        "ConvLSTMCell",
        "ConvLSTMEncoder",
        "SpatialEncoder",
        "TemporalInceptionBlock",
        "TemporalTranslator",
        "SpatialDecoder",
    )
    for class_name in class_names:
        with self.subTest(class_name=class_name):
            self.assertEqual(
                class_ast(MODEL_PATH, class_name),
                class_ast(BASELINE_MODEL_PATH, class_name),
            )

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
    banned_calls = {
        "open", "eval", "exec", "compile", "__import__",
        "system", "popen", "Popen", "run",
    }
    self.assertLessEqual(import_roots, {"torch"})
    self.assertFalse(called_names & banned_calls)

def test_primary_configuration_parameter_overhead_is_below_fifteen_percent(self):
    joint_config = model_config()
    baseline_config = {
        key: value
        for key, value in joint_config.items()
        if key not in {"gate_hidden_dim", "initial_gate_strength"}
    }
    joint = self.module.build_model(joint_config)
    baseline = self.baseline.build_model(baseline_config)
    overhead = (
        trainable_parameter_count(joint) - trainable_parameter_count(baseline)
    ) / trainable_parameter_count(baseline)
    self.assertLess(overhead, 0.15)
```

- [ ] **Step 2: Run the new checks**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_main_path_classes_are_ast_identical_to_baseline tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_uses_only_upload_safe_imports_and_calls tests.test_convlstm_ls_topography_joint_gated_simvp.LsTopographyJointGatedSimVPTests.test_primary_configuration_parameter_overhead_is_below_fifteen_percent -v
```

Expected: all three tests `PASS`. If the AST check fails, restore the copied
main-path class verbatim rather than weakening the assertion. If the budget
test fails, reduce only `gate_hidden_dim` or the condition-encoder widths; do
not shrink the baseline.

- [ ] **Step 3: Run the complete focused suite**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: all joint-model tests `PASS`.

- [ ] **Step 4: Run every repository test**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: the full existing and new suite `PASS` with no failures or errors.

- [ ] **Step 5: Review the final diff and upload surface**

Run:

```powershell
git diff --check
git diff -- "models/convlstm_ls_topography_joint_gated_simvp.py" "tests/test_convlstm_ls_topography_joint_gated_simvp.py"
```

Expected: no whitespace errors; the model diff contains only the declared
metadata, two condition encoders, one joint gate, validation, integration, and
builder changes.

- [ ] **Step 6: Commit final verification coverage**

```powershell
git add -- "models/convlstm_ls_topography_joint_gated_simvp.py" "tests/test_convlstm_ls_topography_joint_gated_simvp.py"
git commit -m "test: verify LS topography gated model"
```
