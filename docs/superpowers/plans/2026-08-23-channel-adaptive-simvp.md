# Channel-Adaptive ConvLSTM-SimVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone uploaded-model script with horizon-conditioned, channel-wise adaptive temporal fusion between ConvLSTM features and a residual SimVP refiner.

**Architecture:** Encode every history step with ConvLSTM and a shared spatial encoder. Compute normalized `[B, horizon, window, spatial_channels]` attention, fuse history features per forecast step, refine them with SimVP blocks, and decode each forecast frame.

**Tech Stack:** Python, PyTorch, standard-library `unittest`

---

### Task 1: Contract and attention tests

**Files:**
- Create: `tests/test_convlstm_adaptive_simvp.py`

- [x] Add a loader for `models/convlstm_adaptive_simvp.py` and assert the complete `MODEL_SPEC` parameter schema.
- [x] Assert `ChannelTemporalFusion.attention_weights` returns `[B, K, T, S]`, sums to one over `T`, remains finite, and has a nonzero finite derivative with respect to encoded inputs.
- [x] Assert platform and odd multichannel inputs produce exact output shapes and finite gradients for every trainable stage.
- [x] Parse the upload AST and reject imports outside `torch` or documented banned calls.
- [x] Run `python -B -m unittest discover -s tests -p "test_convlstm_adaptive_simvp.py" -v`; expect one clear failure because the model file is absent.

### Task 2: Adaptive fusion model

**Files:**
- Create: `models/convlstm_adaptive_simvp.py`
- Test: `tests/test_convlstm_adaptive_simvp.py`

- [x] Add `MODEL_SPEC` with `convlstm_hidden_dim`, `spatial_hidden_dim`, `temporal_hidden_dim`, `num_temporal_blocks`, `dropout`, and `attention_initial_strength`.
- [x] Implement the ConvLSTM cell/encoder and shared stride-two spatial encoder.
- [x] Implement `ChannelTemporalFusion`: pool and normalize space, multiply projected keys by learned `[K,S]` forecast queries, add `[T,S]` history bias, project channel scores without bias, softmax over history, blend with a uniform prior, and fuse with `torch.einsum("bkts,btshw->bkshw", weights, encoded)`.
- [x] Implement a residual SimVP refiner using normalized depthwise `3x3`/`5x5` branches and pointwise fusion.
- [x] Decode forecast features, interpolate only for odd-size alignment, and return `[B,K,1,H,W]`.
- [x] Validate positive integer configuration, `window >= 2`, dropout in `[0,1)`, and attention initialization in `(0,1)`.

### Task 3: Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-08-23-channel-adaptive-simvp.md`

- [x] Run the focused adaptive-model tests and the complete repository test suite.
- [x] Run an independent platform-style UTF-8 parse, import, dry forward pass, finite-output check, and full finite-gradient check.
- [x] Review the new script against the approved design and preserve all unrelated workspace changes.
