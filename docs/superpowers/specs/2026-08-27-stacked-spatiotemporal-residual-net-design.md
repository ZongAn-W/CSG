# Stacked Spatiotemporal Residual Net Design

## Goal

Create a standalone AresVision uploaded-model script that implements the
three-part architecture diagram as one reusable spatiotemporal residual block.
The complete model stacks a configurable number of these blocks, defaults to
three blocks, supports arbitrary configured history and forecast lengths, and
returns the platform-required prediction shape.

## Platform Contract

The script will be created at
`models/stacked_spatiotemporal_residual_net.py`. It will import only `torch`
and export:

- `MODEL_SPEC`, containing model metadata and the adjustable parameter schema.
- `build_model(config)`, returning a `torch.nn.Module`.

The model accepts:

```text
[batch, window, in_channels, height, width]
```

and returns:

```text
[batch, horizon, 1, height, width]
```

It does not declare or accept Ls or topography auxiliary inputs.

## Configurable Parameters

`MODEL_SPEC["parameters"]` exposes:

| Parameter | Type | Default | Range | Meaning |
| --- | --- | ---: | ---: | --- |
| `hidden_dim` | `int` | 32 | 4-128 | Per-time-step feature channels |
| `spatial_dim` | `int` | 128 | 8-512 | Internal multiscale branch channels |
| `num_blocks` | `int` | 3 | 1-8 | Number of complete stacked blocks |
| `dropout` | `float` | 0.1 | 0.0-0.9 | Spatial dropout probability |

The platform-supplied `in_channels`, `window`, and `horizon` values determine
the input stem, temporal projections, and prediction shape.

## Complete Model Architecture

1. Apply a shared padded `3x3` input convolution and GELU independently to
   every historical frame, mapping `in_channels` to `hidden_dim` while
   preserving the spatial dimensions.
2. Pass `[B, window, hidden_dim, H, W]` through `num_blocks` complete
   spatiotemporal residual blocks. Every block preserves this shape.
3. Flatten the history and feature dimensions, then apply a learned `1x1`
   projection from `window * hidden_dim` channels to
   `horizon * hidden_dim` channels.
4. Reshape to `[B, horizon, hidden_dim, H, W]` and apply one shared padded
   `3x3` prediction head to every forecast step, producing one output channel.

The time projection makes `window` and `horizon` independent positive
integers rather than requiring both to equal three.

## Complete Block Architecture

Each block receives `X` with shape `[B, window, hidden_dim, H, W]`.

### Two-Layer ConvLSTM

A two-layer ConvLSTM processes the full history at native spatial resolution.
Each layer has `hidden_dim` hidden channels and padded `3x3` recurrent gates.
Hidden and cell states start at zero for every `forward` call and are not
retained across batches.

The second-layer output sequence is added to the block input:

```text
R = X + ConvLSTM2(X)
```

This is the temporal residual, and `R` retains shape
`[B, window, hidden_dim, H, W]`.

### Time-to-Channel Fusion

Reshape `R` to:

```text
[B, window * hidden_dim, H, W]
```

This is a lossless layout change. It exposes all historical feature groups to
ordinary two-dimensional channel mixing.

### Multiscale Spatial Residual Path

1. Apply a `1x1` input projection from `window * hidden_dim` to
   `spatial_dim`, producing `F0`.
2. Feed `F0` to three parallel standard convolutions:
   `3x3`, `5x5`, and `7x7`, each mapping `spatial_dim` to `spatial_dim`.
3. Concatenate the three branch results into `3 * spatial_dim` channels.
4. Fuse them with a `1x1` convolution back to `spatial_dim`, followed by
   `GroupNorm(1, spatial_dim)` and `Dropout2d(dropout)`.
5. Add the fused result to `F0` and apply GELU. This is the internal spatial
   residual.
6. Apply a `1x1` output projection from `spatial_dim` to
   `window * hidden_dim`.
7. Add the flattened `R` as the outer spatial residual.
8. Reshape the result back to `[B, window, hidden_dim, H, W]`.

All convolutions use padding that preserves `H` and `W`. The multiscale
branches are full standard convolutions, not depthwise-separable substitutes.

## Residual Hierarchy

Every complete block contains three residual paths:

1. A temporal residual around the two-layer ConvLSTM.
2. An internal `spatial_dim` residual around multiscale fusion.
3. An outer `window * hidden_dim` residual around the entire spatial path.

This hierarchy follows the supplied diagram while ensuring every block has an
identical input and output shape and can therefore be stacked directly.

## Validation And Errors

`build_model(config)` will reject booleans and non-positive values for integer
fields. It will require `dropout` to be numeric and in `[0, 1)`. Runtime
validation will reject inputs that are not rank five or whose actual window or
channel count differs from the build configuration.

The use of `GroupNorm(1, spatial_dim)` permits every allowed `spatial_dim`
without imposing an additional divisibility constraint.

## Verification

Tests will verify:

- The model file, `MODEL_SPEC`, and `build_model(config)` contract.
- The platform dry-run mapping from `[2, 3, 1, 8, 16]` to
  `[2, 3, 1, 8, 16]` with finite outputs.
- Different `window` and `horizon` values, multiple input channels, odd spatial
  dimensions, and complete finite backpropagation.
- The configured block count, two ConvLSTM layers per block, and standard
  `3x3`, `5x5`, and `7x7` multiscale branches.
- The three shape-compatible residual paths.
- Invalid build parameters and invalid runtime tensor shapes.
- Upload-safe imports and the absence of prohibited calls.

## Scope

This work adds one independent upload script and its focused test file. It
does not modify existing model scripts, add auxiliary datasets, introduce
attention or gating mechanisms, or change the AresVision training platform.
