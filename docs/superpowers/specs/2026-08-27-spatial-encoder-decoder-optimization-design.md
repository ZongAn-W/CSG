# Spatial Encoder Decoder Optimization Design

## Goal

Reduce the training cost of the stacked spatiotemporal residual model by
running ConvLSTM and multiscale spatial blocks on a half-resolution latent
grid, while preserving the AresVision input/output contract and odd-size
support.

## Data Flow

```text
[B, window, in_channels, H, W]
  -> shared SpatialEncoder, stride 2
[B, window, hidden_dim, ceil(H/2), ceil(W/2)]
  -> num_blocks spatiotemporal residual blocks
  -> window-to-horizon 1x1 projection
[B, horizon, hidden_dim, ceil(H/2), ceil(W/2)]
  -> shared SpatialDecoder
[B, horizon, 1, H, W]
```

The encoder is applied independently to each history frame before the first
residual Block, so both ConvLSTM and all `3x3/5x5/7x7` branches operate at
reduced resolution. The decoder is applied independently to each forecast
latent frame after the time projection.

## Spatial Encoder

`SpatialEncoder` uses a padded `3x3` convolution with `stride=2`, mapping
`in_channels` to `hidden_dim`, followed by GELU and a padded `3x3` refinement
convolution that keeps `hidden_dim`. The latent dimensions are
`ceil(H/2)` and `ceil(W/2)` for arbitrary positive spatial sizes.

## Spatial Decoder

`SpatialDecoder` uses a `ConvTranspose2d` with kernel 4, stride 2, and padding
1, followed by GELU and a padded `3x3` convolution from `hidden_dim` to one
output channel. Because transposed convolution can produce one extra pixel for
odd latent dimensions, the decoder interpolates to the exact requested
`(H, W)` whenever needed.

## Unchanged Components

The two-layer ConvLSTM, three residual paths, full standard multiscale branches,
configurable `num_blocks`, `hidden_dim`, `spatial_dim`, and `dropout`, and the
`window` to `horizon` pointwise time projection remain unchanged. No auxiliary
inputs or unsafe imports are introduced.

## Verification

Tests will assert stride-two encoder behavior, decoder exact-size restoration
for even and odd shapes, latent Block execution, platform output shape,
general window/horizon and multichannel backpropagation, and the existing
upload-safety and validation guarantees.
