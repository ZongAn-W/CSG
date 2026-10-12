# Earth Three-Hour SimVP

Upload `earth_3hourly_simvp.py` as a standalone model for
`earth_merra2_3hourly_v1`. The file exports `MODEL_SPEC` and
`build_model(config)` and depends only on PyTorch.

## Architecture

This is a lightweight SimVP variant using the repository's existing
multiscale translator design:

1. A shared two-layer spatial encoder processes every history frame,
   reducing each 24 x 48 tile to 12 x 24 latent features.
2. Time and latent channels are folded together. A 1 x 1 projection mixes
   the entire history, followed by residual Inception blocks with depthwise
   3 x 3 and 5 x 5 branches. Another projection produces the requested
   number of future latent frames.
3. A shared transposed-convolution decoder restores each future frame to
   24 x 48 and projects it to one TO3 channel.

Input is normalized float32 `[B, window, C, 24, 48]`. Output is normalized
float32 `[B, horizon, 1, 24, 48]`, with no final activation restricting
negative standardized values. Window and horizon may differ; both support
1 through 240 steps. The channel order is TO3 followed by the selected
U10M, V10M, T2M, SWGDN auxiliaries in that order.

GroupNorm operates independently on each sample. Dropout is active only
in training, and evaluation does not update parameters or buffers.

## Parameters

| Parameter | Default | Range |
| --- | --- | --- |
| `spatial_hidden_dim` | 32 | 4-256 |
| `temporal_hidden_dim` | 64 | 8-512 |
| `num_temporal_blocks` | 3 | 1-8 |
| `dropout` | 0.1 | 0.0-0.9 |

The factory fills omitted custom parameters from these defaults. The
server supplies `in_channels`, `window`, `horizon`, and dataset metadata.
The standard 56-to-24, five-channel configuration has 224,161 parameters.
Large hidden dimensions and long windows increase memory and compute.

The platform owns loading, task splits, normalization, tile stitching,
training, metrics, and checkpoints. The model receives only `x`; it has
no coordinates, separate time inputs, or cross-tile context. See the
[upload contract](../模板/aresvision_earth_3hourly_model_v1.md).

## Local Verification

Run `python -m unittest discover -s tests -p test_earth_3hourly_simvp.py -v`.
These tests check the tensor contract, training gradients, evaluation
independence, and state-dict reloads on synthetic data. Platform upload
validation and real-data training remain separate checks.
