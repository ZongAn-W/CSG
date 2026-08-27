# Channel-Adaptive ConvLSTM-SimVP Design

## Goal

Create a second AresVision-compatible uploaded model that preserves the existing ConvLSTM-SimVP baseline and replaces fixed input-time concatenation with horizon-conditioned, channel-wise adaptive temporal fusion.

## Architecture

The model processes `[B, T, C, H, W]` through ConvLSTM and a shared spatial encoder to obtain `E` with shape `[B, T, S, h, w]`. A channel temporal fusion module spatially pools and normalizes `E`, multiplies sample-dependent keys by learned forecast queries, adds history-position biases, and normalizes scores over `T` to produce weights `[B, K, T, S]`. A fixed inverse temperature of `2.0` keeps query gradients material at initialization.

The adaptive weights are blended with a uniform temporal prior through a learned per-channel gate initialized from `attention_initial_strength`. Weighted history features produce `[B, K, S, h, w]`. A residual SimVP temporal refiner processes the flattened `[B, K*S, h, w]` representation before the shared decoder returns `[B, K, 1, H, W]`.

## Upload Interface

The single upload file exports `MODEL_SPEC`, `build_model(config)`, and all required `torch.nn.Module` classes. It imports only `torch`. Adjustable parameters are the five baseline capacity parameters plus `attention_initial_strength`, bounded to `(0, 1)` by the schema and builder validation. Because adaptive temporal attention is untrainable with one history step, the builder requires `window >= 2`.

## Stability

For adaptive attention `a` and learned channel gate `g`, the effective weights are `(1-g)/T + g*a`. This keeps initialization close to uniform fusion while allowing every channel to learn its own degree of temporal selectivity. The residual temporal refiner preserves the fused representation when its learned correction is small.

## Verification

Tests cover full schema validity, platform dry-run shape, attention normalization, horizon/channel-specific attention shape, input-dependent attention gradients, odd spatial sizes, multichannel inputs, end-to-end finite gradients, invalid configuration, runtime shape validation, and the platform import/call allowlist.
