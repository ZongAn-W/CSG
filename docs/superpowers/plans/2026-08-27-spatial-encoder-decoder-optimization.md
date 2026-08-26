# Spatial Encoder Decoder Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a stride-two Spatial Encoder and exact-size Spatial Decoder so all recurrent and multiscale Block computation runs on a half-resolution latent grid.

**Architecture:** Encode each history frame with a shared stride-two `3x3` stem plus refinement convolution, run the existing shape-preserving Blocks at latent resolution, project history channels to forecast channels, and decode each forecast latent frame with a transposed convolution and final size alignment. The output contract remains `[B,horizon,1,H,W]` for even and odd dimensions.

**Tech Stack:** Python, PyTorch, standard-library `unittest`

---

### Task 1: Test Encoder And Decoder Contracts First

**Files:**
- Modify: `tests/test_stacked_spatiotemporal_residual_net.py`
- Modify: `models/stacked_spatiotemporal_residual_net.py`

- [x] **Step 1: Write failing tests**

Add tests that instantiate `SpatialEncoder(2, 4)` and assert its first
convolution has `stride == (2, 2)` and that input `[1,2,9,15]` produces
`[1,4,5,8]`. Instantiate `SpatialDecoder(4)` and assert decoding a latent
`[1,4,5,8]` with `output_size=(9,15)` returns `[1,1,9,15]`.

- [x] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_spatial_encoder_downsamples_by_two tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_spatial_decoder_restores_requested_size -v
```

Expected: `ERROR` because `SpatialEncoder` and `SpatialDecoder` are not yet
defined.

- [x] **Step 3: Implement the minimal encoder and decoder**

Add:

```python
class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class SpatialDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, output_size):
        x = self.layers(x)
        if x.shape[-2:] != output_size:
            x = torch.nn.functional.interpolate(
                x, size=output_size, mode="bilinear", align_corners=False
            )
        return x
```

- [x] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2. Expected: both tests `PASS`.

### Task 2: Route The Complete Model Through Latent Space

**Files:**
- Modify: `tests/test_stacked_spatiotemporal_residual_net.py`
- Modify: `models/stacked_spatiotemporal_residual_net.py`

- [x] **Step 1: Write failing integration assertions**

Extend the dry-run test to assert `model.spatial_encoder.layers[0].stride ==
(2, 2)` and `isinstance(model.spatial_decoder, module.SpatialDecoder)`. Extend
the general odd-size test to keep requiring output `[1,2,1,9,15]` and finite
gradients.

- [x] **Step 2: Run integration tests and verify RED**

Run:

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_matches_platform_dry_run_shape tests.test_stacked_spatiotemporal_residual_net.StackedSpatiotemporalResidualNetTests.test_supports_general_time_channel_and_spatial_shapes -v
```

Expected: `ERROR` because the current model has no encoder/decoder attributes.

- [x] **Step 3: Replace the full-resolution stem/head data flow**

In `StackedSpatiotemporalResidualNet.__init__`, replace the input projection
and activation with:

```python
self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim)
self.spatial_decoder = SpatialDecoder(hidden_dim)
```

Keep the existing Blocks and time projection. In `forward`, encode reshaped
history frames, capture latent height/width, reshape to `[B,window,hidden_dim,
latent_height,latent_width]`, run Blocks and time projection at that latent
size, decode reshaped forecast frames with `output_size=(height,width)`, and
reshape to `[B,horizon,1,height,width]`.

- [x] **Step 4: Run integration tests and verify GREEN**

Run the command from Step 2. Expected: both tests `PASS` for even and odd
spatial sizes and finite gradients.

### Task 3: Regression And Safety Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-08-27-spatial-encoder-decoder-optimization.md`

- [x] **Step 1: Run focused tests**

```powershell
python -m unittest tests.test_stacked_spatiotemporal_residual_net -v
```

Expected: all focused tests `PASS`.

- [x] **Step 2: Run full regression and static checks**

```powershell
python -m unittest discover -s tests -v
python -m py_compile models/stacked_spatiotemporal_residual_net.py tests/test_stacked_spatiotemporal_residual_net.py
git diff --check
```

Expected: all tests pass, compilation exits zero, and diff check is clean.

- [x] **Step 3: Mark this plan complete and commit the optimization**

Mark all checklist items `[x]`, then stage only the updated model, focused
tests, optimization spec, and this plan:

```powershell
git add -- models/stacked_spatiotemporal_residual_net.py tests/test_stacked_spatiotemporal_residual_net.py docs/superpowers/specs/2026-08-27-spatial-encoder-decoder-optimization-design.md docs/superpowers/plans/2026-08-27-spatial-encoder-decoder-optimization.md
git commit -m "perf: add latent spatial encoder and decoder"
```
