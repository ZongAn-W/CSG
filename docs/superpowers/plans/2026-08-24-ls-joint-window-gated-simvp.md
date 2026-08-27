# Ls-Conditioned Joint Window-Gated SimVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-file AresVision model that jointly computes independent Ls-conditioned weights for every input window while retaining the C-S-1 main path.

**Architecture:** ConvLSTM and the shared spatial encoder produce `[B,T,S,h,w]`. A joint MLP consumes all pooled window descriptors and per-step Ls harmonics, then independently sigmoid-gates residual scales before the unchanged C-S-1 translator and decoder.

**Tech Stack:** Python, PyTorch, `unittest`, Python AST.

---

### Task 1: Establish The Upload Contract Red Test

**Files:**
- Create: `tests/test_convlstm_ls_joint_window_gated_simvp.py`

- [ ] **Step 1: Create the test loader and missing-file assertion**

```python
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "convlstm_ls_joint_window_gated_simvp.py"

def test_model_file_exists(self):
    self.assertTrue(
        MODEL_PATH.is_file(),
        "missing model file: models/convlstm_ls_joint_window_gated_simvp.py",
    )
```

Add skipped-until-present tests for the full schema, forward shapes, gate
behavior, input validation, gradients, AST safety, and baseline equivalence.

- [ ] **Step 2: Run the isolated test file**

Run:

```powershell
python -m unittest discover -s tests -p 'test_convlstm_ls_joint_window_gated_simvp.py' -v
```

Expected: one failure containing
`missing model file: models/convlstm_ls_joint_window_gated_simvp.py`; remaining
tests skip because the model file is absent.

### Task 2: Implement The Standalone Model

**Files:**
- Create: `models/convlstm_ls_joint_window_gated_simvp.py`
- Test: `tests/test_convlstm_ls_joint_window_gated_simvp.py`

- [ ] **Step 1: Copy the C-S-1 main-path classes unchanged**

Copy `ConvLSTMCell`, `ConvLSTMEncoder`, `SpatialEncoder`,
`TemporalInceptionBlock`, `TemporalTranslator`, and `SpatialDecoder` byte-level
equivalently from `models/convlstm_simvp.py` so AST comparison succeeds.

- [ ] **Step 2: Implement the joint window gate**

```python
descriptors = encoded.mean(dim=(-2, -1))
harmonics = self.build_harmonic_features(ls)
joint_input = torch.cat((descriptors, harmonics), dim=-1).reshape(batch, -1)
weights = torch.sigmoid(self.scorer(joint_input))
strength = torch.sigmoid(self.strength_logit)
scales = 1.0 + strength * (2.0 * weights - 1.0)
weighted = scales.reshape(batch, window, 1, 1, 1) * encoded
```

Use a dense `Linear -> GELU -> Linear` scorer so every output can depend on all
windows. Initialize the output weight with Xavier gain `0.01`, output bias to
zero, and the bounded strength logit from `initial_window_strength`.

- [ ] **Step 3: Implement the AresVision model and schema**

Export exact Ls auxiliary metadata, the five baseline parameters,
`window_gate_hidden_dim`, and `initial_window_strength`. Validate `x`, `ls`,
configuration types/ranges, shape, floating dtype, finite values, and matching
devices. Reshape scaled features to `[B,T*S,h,w]` before the unchanged temporal
translator and return `[B,horizon,1,H,W]`.

- [ ] **Step 4: Run the isolated tests until green**

Run:

```powershell
python -m unittest discover -s tests -p 'test_convlstm_ls_joint_window_gated_simvp.py' -v
```

Expected: every new test passes with no warnings.

### Task 3: Regression And Experimental-Hygiene Verification

**Files:**
- Verify: `models/convlstm_ls_joint_window_gated_simvp.py`
- Verify: `tests/test_convlstm_ls_joint_window_gated_simvp.py`

- [ ] **Step 1: Run the complete suite**

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: zero failures, errors, skips, or warnings.

- [ ] **Step 2: Produce diagnostics**

Build the default model with `horizon=20` and report parameter count versus
C-S-1, output shape, window weights and scales, strength, and gradient counts
for the joint gate. Confirm AST imports include only `torch` and that no
existing model source was edited.
