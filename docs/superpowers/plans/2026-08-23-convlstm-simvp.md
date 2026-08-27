# ConvLSTM-SimVP Uploaded Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-file uploaded model that feeds a full ConvLSTM hidden sequence into a lightweight SimVP encoder-translator-decoder and satisfies the AresVision tensor contract.

**Architecture:** A ConvLSTM encodes every input time step at full resolution. A shared spatial encoder downsamples each hidden state, an inception-style residual translator maps the flattened input-time channels to forecast-time channels, and a shared decoder reconstructs one output channel per forecast step.

**Tech Stack:** Python, PyTorch, standard-library `unittest`

---

### Task 1: Define the uploaded-model contract tests

**Files:**
- Create: `tests/test_convlstm_simvp.py`

- [x] **Step 1: Write a loader and metadata test**

Load `models/convlstm_simvp.py` with `importlib.util`, assert `MODEL_SPEC` exposes all five adjustable parameters, and assert `build_model(config)` returns an `nn.Module`.

- [x] **Step 2: Write tensor-contract tests**

Build with the platform dry-run configuration and assert `(2, 3, 1, 8, 16)` maps to `(2, 3, 1, 8, 16)`. Add a second case using two input channels, `window=4`, `horizon=2`, and odd `9x15` spatial dimensions; backpropagate the output mean and assert trainable gradients exist.

- [x] **Step 3: Write validation and upload-safety tests**

Assert a mismatched runtime window raises `ValueError`. Parse the model AST and assert imports are rooted at `torch` and no banned call name (`open`, `eval`, `exec`, `compile`, `__import__`, `system`, `popen`, `Popen`, or `run`) occurs.

- [x] **Step 4: Run the tests to verify RED**

Run: `python -m unittest discover -s tests -p "test_convlstm_simvp.py" -v`

Expected: FAIL because `models/convlstm_simvp.py` does not exist.

### Task 2: Implement the ConvLSTM feature encoder

**Files:**
- Create: `models/convlstm_simvp.py`
- Test: `tests/test_convlstm_simvp.py`

- [x] **Step 1: Declare the upload schema**

Export `MODEL_SPEC` with integer parameters `convlstm_hidden_dim`, `spatial_hidden_dim`, `temporal_hidden_dim`, `num_temporal_blocks`, plus float parameter `dropout`, all with bounded defaults.

- [x] **Step 2: Implement `ConvLSTMCell`**

Concatenate the current frame and previous hidden state, compute four gates with one padded `3x3` convolution, update the cell state, and return the hidden/cell pair. Initialize recurrent state with `x.new_zeros` so dtype and device follow the input.

- [x] **Step 3: Implement `ConvLSTMEncoder`**

Iterate over `x[:, step]`, retain the state between steps, collect every hidden state, and return `torch.stack(outputs, dim=1)` with shape `[B, window, hidden, H, W]`.

### Task 3: Implement the lightweight SimVP stages

**Files:**
- Modify: `models/convlstm_simvp.py`
- Test: `tests/test_convlstm_simvp.py`

- [x] **Step 1: Implement the shared spatial encoder**

Use a stride-two `3x3` convolution followed by GELU and a stride-one `3x3` convolution. Apply it to the flattened `[B*window, hidden, H, W]` tensor.

- [x] **Step 2: Implement inception-style temporal residual blocks**

Normalize with `GroupNorm(1, channels)`, process with parallel depthwise `3x3` and `5x5` branches, concatenate, merge through a pointwise convolution, apply GELU/dropout, and add the input residual.

- [x] **Step 3: Map input time to forecast time**

Reshape encoded features to `[B, window*spatial_hidden, h, w]`, project to `temporal_hidden`, apply the configured residual blocks, and project to `[B, horizon*spatial_hidden, h, w]`.

- [x] **Step 4: Decode forecast frames**

Decode `[B*horizon, spatial_hidden, h, w]` with a stride-two transposed convolution and an output convolution. Use bilinear interpolation only when necessary to match the original height and width, then reshape to `[B, horizon, 1, H, W]`.

### Task 4: Build, validate, and verify

**Files:**
- Modify: `models/convlstm_simvp.py`
- Test: `tests/test_convlstm_simvp.py`

- [x] **Step 1: Implement `ConvLSTMSimVP.forward`**

Reject tensors that are not rank five or whose window/channel counts differ from the build configuration, then execute the ConvLSTM-to-SimVP data flow.

- [x] **Step 2: Implement `build_model(config)`**

Read platform and custom keys, validate every integer is positive and dropout lies in `[0, 1)`, and return `ConvLSTMSimVP`.

- [x] **Step 3: Run focused tests to verify GREEN**

Run: `python -m unittest discover -s tests -p "test_convlstm_simvp.py" -v`

Expected: all tests pass.

- [x] **Step 4: Run final syntax and platform dry-run checks**

Parse the file as UTF-8 Python, load it as a module, call `build_model` with the documented defaults, and verify the exact dry-run output shape is `(2, 3, 1, 8, 16)`.
