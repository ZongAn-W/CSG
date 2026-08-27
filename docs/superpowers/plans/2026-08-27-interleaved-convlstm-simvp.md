# Interleaved ConvLSTM-SimVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone forecasting model that applies a ConvLSTM between every SimVP spatial block without modifying the existing ConvLSTM-SimVP baseline.

**Architecture:** Encode every history frame spatially, then project it into a shared temporal channel space. Each interleaved block runs an independent ConvLSTM over the full history window, projects its hidden states back to the temporal channels through a residual connection, and applies the SimVP multiscale residual block to every frame. A learned pointwise projection folds the final history features into the requested forecast horizon before spatial decoding.

**Tech Stack:** Python, PyTorch, unittest.

---

### Task 1: Specify the interleaved model contract

**Files:**
- Create: `tests/test_interleaved_convlstm_simvp.py`

- [ ] **Step 1: Write the failing test**

```python
def test_each_simvp_block_contains_and_executes_a_convlstm():
    model = self.module.build_model(model_config(num_temporal_blocks=2)).eval()
    calls = [0, 0]
    hooks = [
        block.convlstm.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, block in enumerate(model.temporal_translator.blocks)
    ]
    try:
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 1, 8, 16))
    finally:
        for hook in hooks:
            hook.remove()

    self.assertEqual(tuple(outputs.shape), (2, 3, 1, 8, 16))
    self.assertEqual(calls, [1, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_interleaved_convlstm_simvp.InterleavedConvLSTMSimVPTests.test_each_simvp_block_contains_and_executes_a_convlstm -v`

Expected: FAIL because `models/interleaved_convlstm_simvp.py` does not exist.

### Task 2: Implement the standalone interleaved model

**Files:**
- Create: `models/interleaved_convlstm_simvp.py`
- Test: `tests/test_interleaved_convlstm_simvp.py`

- [ ] **Step 1: Add the model components**

```python
class InterleavedTemporalBlock(nn.Module):
    def __init__(self, channels, convlstm_hidden_dim, dropout):
        super().__init__()
        self.convlstm = ConvLSTMEncoder(channels, convlstm_hidden_dim)
        self.state_projection = nn.Conv2d(convlstm_hidden_dim, channels, 1)
        self.simvp_block = TemporalInceptionBlock(channels, dropout)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        recurrent = self.convlstm(x).reshape(
            batch * steps, self.convlstm.hidden_dim, height, width
        )
        residual = self.state_projection(recurrent).reshape(
            batch, steps, channels, height, width
        )
        features = (x + residual).reshape(batch * steps, channels, height, width)
        return self.simvp_block(features).reshape(batch, steps, channels, height, width)
```

- [ ] **Step 2: Add model assembly and input validation**

```python
encoded = self.spatial_encoder(x.reshape(batch * window, channels, height, width))
encoded = self.input_projection(encoded).reshape(
    batch, window, self.temporal_hidden_dim, encoded_height, encoded_width
)
translated = self.temporal_translator(encoded)
forecast = self.output_projection(
    translated.reshape(batch, window * self.temporal_hidden_dim, encoded_height, encoded_width)
)
```

- [ ] **Step 3: Run the targeted test to verify it passes**

Run: `python -m unittest tests.test_interleaved_convlstm_simvp.InterleavedConvLSTMSimVPTests.test_each_simvp_block_contains_and_executes_a_convlstm -v`

Expected: PASS.

### Task 3: Cover integration and safety behavior

**Files:**
- Modify: `tests/test_interleaved_convlstm_simvp.py`
- Test: `tests/test_interleaved_convlstm_simvp.py`

- [ ] **Step 1: Add real-forward tests**

```python
def test_supports_multichannel_odd_sized_inputs_and_backpropagation(self):
    model = self.module.build_model(model_config(in_channels=2, window=4, horizon=2))
    outputs = model(torch.randn(1, 4, 2, 9, 15))
    outputs.square().mean().backward()
    self.assertEqual(tuple(outputs.shape), (1, 2, 1, 9, 15))
```

- [ ] **Step 2: Add contract, invalid-input, and upload-safety tests**

```python
def test_rejects_runtime_window_mismatch(self):
    model = self.module.build_model(model_config())
    with self.assertRaisesRegex(ValueError, "window"):
        model(torch.randn(1, 2, 1, 8, 16))
```

- [ ] **Step 3: Run all model tests**

Run: `python -m unittest tests.test_interleaved_convlstm_simvp -v`

Expected: PASS with all contract, execution, gradient, validation, and AST upload-safety tests passing.

- [ ] **Step 4: Compile and inspect the focused diff**

Run: `python -m py_compile models/interleaved_convlstm_simvp.py`

Run: `git diff --no-index -- /dev/null models/interleaved_convlstm_simvp.py`

Expected: compilation exits with code 0 and the diff contains only the new standalone model.
