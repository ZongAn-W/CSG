# Stacked Spatiotemporal Residual Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an AresVision-compatible single-file PyTorch model that stacks a configurable number of the approved two-layer ConvLSTM and multiscale residual blocks.

**Architecture:** A shared input stem maps every history frame to `hidden_dim`. Each shape-preserving block applies a two-layer ConvLSTM temporal residual, losslessly folds time into channels, applies full `3x3/5x5/7x7` multiscale spatial fusion with two more residual paths, and restores the time layout. A learned pointwise projection maps arbitrary `window` features to arbitrary `horizon` features before a shared output head predicts one channel per lead.

**Tech Stack:** Python, PyTorch, standard-library `unittest`

**Implementation Status:** Complete. The model, focused tests, full regression
suite, syntax compilation, and upload-safety checks have all been verified.

---

## File Structure

- Create `models/stacked_spatiotemporal_residual_net.py`: complete standalone upload model, parameter schema, validation, recurrent cells, reusable block, stacked network, and `build_model`.
- Create `tests/test_stacked_spatiotemporal_residual_net.py`: platform contract, architecture, residual, general-shape, gradient, validation, and upload-safety tests.

### Task 1: Establish The Upload Contract

**Files:**
- Create: `tests/test_stacked_spatiotemporal_residual_net.py`
- Create: `models/stacked_spatiotemporal_residual_net.py`

- [x] **Step 1: Write the failing file and upload-contract tests**

Create a test module that loads the production file by path and defines this
configuration:

```python
def model_config(**overrides):
    config = {
        "in_channels": 1,
        "window": 3,
        "horizon": 3,
        "height": 8,
        "width": 16,
        "selected_channels": [0],
        "hidden_dim": 32,
        "spatial_dim": 128,
        "num_blocks": 3,
        "dropout": 0.1,
    }
    config.update(overrides)
    return config
```

Assert that the file exists, `MODEL_SPEC["parameters"]` exactly matches the
approved four-field schema, and `build_model(model_config())` returns an
`nn.Module`.

- [x] **Step 2: Run the existence test and verify RED**

Run:

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_model_file_exists -v
```

Expected: `FAIL` because the model file is absent.

- [x] **Step 3: Add the model file and upload metadata**

Start the production file with only safe imports and the exact schema:

```python
import torch
from torch import nn


MODEL_SPEC = {
    "name": "StackedSpatiotemporalResidualNet",
    "description": (
        "Stacked two-layer ConvLSTM and full multiscale spatial residual blocks."
    ),
    "parameters": {
        "hidden_dim": {"type": "int", "default": 32, "min": 4, "max": 128},
        "spatial_dim": {"type": "int", "default": 128, "min": 8, "max": 512},
        "num_blocks": {"type": "int", "default": 3, "min": 1, "max": 8},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
    },
}
```

- [x] **Step 4: Run the existence test and verify GREEN**

Run the command from Step 2. Expected: `PASS`.

### Task 2: Implement And Verify One Complete Block

**Files:**
- Modify: `tests/test_stacked_spatiotemporal_residual_net.py`
- Modify: `models/stacked_spatiotemporal_residual_net.py`

- [x] **Step 1: Write failing architecture and residual tests**

The tests will instantiate a small block and assert:

```python
block = module.SpatiotemporalResidualBlock(
    window=3, hidden_dim=4, spatial_dim=8, dropout=0.0
)
self.assertEqual(len(block.temporal.layers), 2)
self.assertEqual(block.branch_3x3.kernel_size, (3, 3))
self.assertEqual(block.branch_5x5.kernel_size, (5, 5))
self.assertEqual(block.branch_7x7.kernel_size, (7, 7))
self.assertEqual(block.branch_3x3.groups, 1)
self.assertEqual(tuple(block(torch.randn(2, 3, 4, 7, 9)).shape), (2, 3, 4, 7, 9))
```

For the residual behavior, zero all ConvLSTM parameters and the spatial output
projection, evaluate the block, and require exact preservation of its input:

```python
for parameter in block.temporal.parameters():
    nn.init.zeros_(parameter)
nn.init.zeros_(block.output_projection.weight)
nn.init.zeros_(block.output_projection.bias)
self.assertTrue(torch.equal(block(inputs), inputs))
```

- [x] **Step 2: Run the block tests and verify RED**

Run:

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_block_matches_approved_architecture tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_zero_updates_preserve_block_input -v
```

Expected: `ERROR` because `SpatiotemporalResidualBlock` is not defined.

- [x] **Step 3: Implement recurrent and multiscale components**

Add:

```python
class ConvLSTMCell(nn.Module):
    # A padded 3x3 convolution produces the four standard LSTM gates.

class StackedConvLSTM(nn.Module):
    # ModuleList of exactly two cells; returns every second-layer hidden state.

class SpatiotemporalResidualBlock(nn.Module):
    # temporal = StackedConvLSTM(hidden_dim, hidden_dim, num_layers=2)
    # temporal residual: recurrent + x
    # fold [B,T,C,H,W] to [B,T*C,H,W]
    # input projection T*C -> spatial_dim
    # full standard 3x3, 5x5, and 7x7 branches
    # concatenate, fuse, GroupNorm(1,...), Dropout2d
    # internal residual then GELU
    # output projection spatial_dim -> T*C
    # outer residual then restore [B,T,C,H,W]
```

The block's forward expression is:

```python
temporal = x + self.temporal(x)
flat = temporal.reshape(batch, window * channels, height, width)
base = self.input_projection(flat)
multiscale = torch.cat(
    (self.branch_3x3(base), self.branch_5x5(base), self.branch_7x7(base)),
    dim=1,
)
fused = self.dropout(self.norm(self.fusion(multiscale)))
spatial = self.activation(base + fused)
output = flat + self.output_projection(spatial)
return output.reshape(batch, window, channels, height, width)
```

- [x] **Step 4: Run the block tests and verify GREEN**

Run the command from Step 2. Expected: both tests `PASS`.

### Task 3: Build The Configurable Complete Model

**Files:**
- Modify: `tests/test_stacked_spatiotemporal_residual_net.py`
- Modify: `models/stacked_spatiotemporal_residual_net.py`

- [x] **Step 1: Write failing complete-model tests**

Add tests for:

```python
model = module.build_model(model_config()).eval()
self.assertEqual(len(model.blocks), 3)
self.assertEqual(tuple(model(torch.randn(2, 3, 1, 8, 16)).shape), (2, 3, 1, 8, 16))
```

and a general shape/backpropagation case:

```python
model = module.build_model(model_config(
    in_channels=2,
    window=4,
    horizon=2,
    selected_channels=[0, 1],
    hidden_dim=4,
    spatial_dim=8,
    num_blocks=2,
    dropout=0.0,
))
outputs = model(torch.randn(1, 4, 2, 9, 15))
outputs.square().mean().backward()
self.assertEqual(tuple(outputs.shape), (1, 2, 1, 9, 15))
```

Require every trainable parameter to have a finite gradient.

- [x] **Step 2: Run the complete-model tests and verify RED**

Run:

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_matches_platform_dry_run_shape tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_supports_general_time_channel_and_spatial_shapes -v
```

Expected: `ERROR` because the complete model and builder are not defined.

- [x] **Step 3: Implement the input stem, stacked blocks, and prediction head**

Implement a model whose central data flow is:

```python
encoded = self.input_activation(
    self.input_projection(x.reshape(batch * window, channels, height, width))
)
encoded = encoded.reshape(batch, window, self.hidden_dim, height, width)
encoded = self.blocks(encoded)
forecast = self.time_projection(
    encoded.reshape(batch, window * self.hidden_dim, height, width)
)
forecast = forecast.reshape(
    batch * self.horizon, self.hidden_dim, height, width
)
prediction = self.output_projection(forecast)
return prediction.reshape(batch, self.horizon, 1, height, width)
```

`self.blocks` is an `nn.Sequential` containing exactly `num_blocks` complete
blocks. `self.time_projection` is a `1x1` convolution from
`window * hidden_dim` to `horizon * hidden_dim`.

- [x] **Step 4: Implement strict build and runtime validation**

Use a `_positive_int(config, key)` helper that rejects booleans and values
less than one. Require numeric `dropout` in `[0, 1)`. In `forward`, reject
non-five-dimensional tensors and mismatched configured window or channel
counts with clear `ValueError` messages.

- [x] **Step 5: Run the complete-model tests and verify GREEN**

Run the command from Step 2. Expected: both tests `PASS` with finite outputs
and gradients.

### Task 4: Complete Validation And Regression Verification

**Files:**
- Modify: `tests/test_stacked_spatiotemporal_residual_net.py`

- [x] **Step 1: Add parameter, runtime, and upload-safety tests**

Assert rejection of zero/boolean integer configuration, `dropout >= 1`, rank
four input, window mismatch, and channel mismatch. Parse the production file
with `ast` and require import roots to be a subset of `{"torch"}` and called
names to exclude:

```python
{
    "open", "eval", "exec", "compile", "__import__",
    "system", "popen", "Popen", "run",
}
```

- [x] **Step 2: Run the focused test module**

Run:

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net -v
```

Expected: all focused tests `PASS`.

- [x] **Step 3: Run the full regression suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all repository tests `PASS`.

- [x] **Step 4: Check source formatting and diff integrity**

Run:

```powershell
python -m py_compile models/stacked_spatiotemporal_residual_net.py tests/test_stacked_spatiotemporal_residual_net.py
git diff --check
```

Expected: both commands exit with code zero and produce no errors.

- [x] **Step 5: Commit the implementation**

Stage only the new model, focused test, and this plan, then commit:

```powershell
git add -- models/stacked_spatiotemporal_residual_net.py tests/test_stacked_spatiotemporal_residual_net.py docs/superpowers/plans/2026-08-27-stacked-spatiotemporal-residual-net.md
git commit -m "feat: add stacked spatiotemporal residual model"
```
