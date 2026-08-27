# Residual-Adaptive ConvLSTM-SimVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third standalone model that keeps the original complete-history SimVP path and adds adaptive temporal attention only as a small residual branch.

**Architecture:** Shared ConvLSTM/spatial features feed both the original `window*channels` SimVP translator and a horizon-conditioned channel-attention adapter. A trainable scalar initialized to `0.05` scales the attention correction before addition to the baseline forecast features.

**Tech Stack:** Python, PyTorch, standard-library `unittest`

---

### Task 1: Write contract and fallback tests

**Files:**
- Create: `tests/test_convlstm_residual_adaptive_simvp.py`

- [x] Assert the target model file and complete seven-parameter schema.
- [x] Assert zero residual scale returns exactly the manually computed baseline-path output.
- [x] Assert the default residual is nonzero, smaller than 25% of the baseline feature norm, and gives the forecast query a finite nonzero gradient.
- [x] Assert standard and odd multichannel input/output contracts plus full finite gradients.
- [x] Assert invalid parameters and runtime shapes fail clearly, and AST imports/calls satisfy upload rules.
- [x] Run the focused test and verify failure because the target model file is absent.

### Task 2: Implement the standalone dual-path model

**Files:**
- Create: `models/convlstm_residual_adaptive_simvp.py`
- Test: `tests/test_convlstm_residual_adaptive_simvp.py`

- [x] Implement the upload schema, ConvLSTM, spatial encoder, inception blocks, original complete-history translator, and decoder.
- [x] Implement normalized multiplicative channel attention with forecast queries, uniform-prior gate, and time-normalized weights.
- [x] Implement an identity-initialized shared adapter and trainable scalar residual scale.
- [x] Combine baseline and residual forecast features, decode, and validate configuration/runtime shapes.
- [x] Run the focused tests until all pass.

### Task 3: Verify the model family

**Files:**
- Modify: `docs/superpowers/plans/2026-08-23-residual-adaptive-simvp.md`

- [x] Run every repository test.
- [x] Run platform-style UTF-8 parsing, import/call scanning, dry forward, exact zero-scale fallback, attention diagnostics, and finite full-gradient checks.
- [x] Review the new model against this design and mark the plan complete.
