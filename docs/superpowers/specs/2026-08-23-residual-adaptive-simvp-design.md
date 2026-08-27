# Residual-Adaptive ConvLSTM-SimVP Design

## Goal

Preserve the complete-history path of the stronger ConvLSTM-SimVP baseline while testing channel-wise temporal attention as a small auxiliary correction instead of an information-replacing bottleneck.

## Architecture

ConvLSTM and the shared spatial encoder produce `E` with shape `[B, T, S, h, w]`. The baseline path reshapes all history features to `[B, T*S, h, w]` and applies the original SimVP translator to produce `[B, K*S, h, w]` without discarding a history step.

In parallel, horizon-conditioned channel attention produces `[B, K, S, h, w]`. A shared identity-initialized `1x1` adapter processes each forecast step. The paths combine as `Z = Z_base + lambda * Z_attention`, where trainable scalar `lambda` starts at the configurable `residual_branch_scale`, default `0.05`. Setting it to zero exactly restores the baseline feature path.

## Interface And Safety

The new standalone upload file is `models/convlstm_residual_adaptive_simvp.py`; the two earlier model files remain unchanged for controlled comparison. It imports only `torch`, exports the required AresVision interface, requires `window >= 2`, and supports odd spatial sizes through output alignment.

## Verification

Tests prove the upload schema, exact zero-scale baseline fallback, nonzero but bounded default residual contribution, adaptive attention behavior and gradients, platform dry-run shape, odd multichannel shapes, full finite backpropagation, validation branches, and upload-safe imports/calls.

