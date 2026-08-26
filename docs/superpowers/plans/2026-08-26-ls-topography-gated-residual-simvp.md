# Ls-Topography Gated Residual SimVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let historical solar longitude and topography add conditional features as well as gate existing features without changing the AresVision input contract.

**Architecture:** Reuse the joint hidden representation already built from encoded dynamics, Ls harmonics, and terrain. Add a separately initialized projection and bounded strength for an additive residual before the unchanged SimVP temporal translator.

**Tech Stack:** Python, PyTorch, `unittest`, Python AST.

---

### Task 1: Define The Conditional Residual Contract

**Files:**
- Modify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`
- Test: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Write failing schema and configuration tests**

Require `initial_residual_strength` in `MODEL_SPEC`, add it to `model_config`,
and assert values outside `[0,1]` are rejected:

```python
self.assertIn("initial_residual_strength", self.module.MODEL_SPEC["parameters"])
with self.assertRaisesRegex(ValueError, "initial_residual_strength"):
    self.module.build_model(model_config(initial_residual_strength=-0.01))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp.ConvLSTMLsTopographyJointGatedSimVPTests.test_exports_baseline_parameters_plus_gate_parameters tests.test_convlstm_ls_topography_joint_gated_simvp.ConvLSTMLsTopographyJointGatedSimVPTests.test_rejects_invalid_gate_configuration -v
```

Expected: failures because the new configuration key is not exported or read.

### Task 2: Specify Additive Residual Behavior

**Files:**
- Modify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`
- Test: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Update the gate return contract test**

Require `(fused, gate_values, scales, residual)` and verify all tensors match
the encoded shape:

```python
fused, gate_values, scales, residual = gate(encoded, ls, topography)
self.assertEqual(residual.shape, encoded.shape)
torch.testing.assert_close(
    fused,
    scales * encoded + gate.current_residual_strength() * residual,
)
```

- [ ] **Step 2: Add a test proving the branch creates information**

Set encoded features to zero, set the residual projection to a deterministic
nonzero mapping, and assert fused output is nonzero. This distinguishes an
additive branch from the old multiplicative-only gate.

- [ ] **Step 3: Add a zero-strength compatibility test**

Build with `initial_residual_strength=0`, reuse shared weights, and assert the
new fusion result equals `scales * encoded` bit-for-bit.

- [ ] **Step 4: Run the three focused tests and verify RED**

Run the named test methods with `python -m unittest ... -v`.

Expected: failures due to missing residual return value, projection, and
strength.

### Task 3: Implement The Minimal Conditional Residual

**Files:**
- Modify: `models/convlstm_ls_topography_joint_gated_simvp.py`
- Test: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Add projection and bounded strength**

In `JointSpatiotemporalGate.__init__`, add:

```python
self.residual_projection = nn.Conv2d(hidden_dim, channels, kernel_size=1)
self.residual_strength = nn.Parameter(torch.tensor(float(initial_residual_strength)))
nn.init.xavier_uniform_(self.residual_projection.weight, gain=0.01)
nn.init.zeros_(self.residual_projection.bias)
```

Add `current_residual_strength()` using the same straight-through clamp as
`current_strength()`.

- [ ] **Step 2: Compute gate and residual from one joint representation**

Refactor the internal helper to form `joint_features` once. Project it through
both heads, and return:

```python
fused = scales * encoded + self.current_residual_strength() * residual
return fused, gate, scales, residual
```

- [ ] **Step 3: Wire the new configuration**

Add `initial_residual_strength` to `MODEL_SPEC`, constructor signatures, and
`build_model`, validating it through `_unit_interval`.

- [ ] **Step 4: Run all dedicated tests until GREEN**

Run:

```powershell
python -m unittest tests.test_convlstm_ls_topography_joint_gated_simvp -v
```

Expected: every dedicated test passes without warnings.

### Task 4: Regression And Diagnostics

**Files:**
- Verify: `models/convlstm_ls_topography_joint_gated_simvp.py`
- Verify: `tests/test_convlstm_ls_topography_joint_gated_simvp.py`

- [ ] **Step 1: Run the complete suite**

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: zero failures and errors.

- [ ] **Step 2: Run source and diff checks**

```powershell
git diff --check
git diff -- models/convlstm_ls_topography_joint_gated_simvp.py tests/test_convlstm_ls_topography_joint_gated_simvp.py
```

Confirm only the intended model and tests changed and no unsafe imports were
introduced.

- [ ] **Step 3: Produce diagnostics**

Report default parameter count and overhead versus the original gated model,
forward output shape, initial gate/residual strengths, branch magnitudes, and
finite gradients for every trainable parameter.
