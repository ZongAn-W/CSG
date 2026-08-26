# Seasonal Topographic Evolution Operator Design

## Goal

Create an independent AresVision uploaded model that tests whether a
seasonally modulated, terrain-relational spatial evolution operator improves
global Mars ozone forecasting over both `convlstm_simvp.py` and
`convlstm_ls_topography_joint_gated_simvp.py`.

The method is named the Seasonal Topographic Evolution Operator (STEO). Its
central hypothesis is that the spatial evolution rule for ozone-related latent
states is non-stationary: the effect associated with a terrain relationship
changes with solar longitude. STEO therefore changes how information moves
between grid cells. It does not merely scale an already encoded feature at each
cell.

The work prioritizes a clear paper-level method contribution over minimal
parameter or runtime overhead. It remains limited to an uploaded model: the
platform data loader, normalization, optimizer, loss, metrics, checkpointing,
and training loop do not change.

## Scientific Claim Boundary

All samples use the same global MOLA field. Terrain is therefore confounded
with geographic position in the observational design. The method may be
described as a seasonal-terrain structural inductive bias, and experiments may
show that predictions and learned operators depend on terrain relationships.
They must not be presented as identifying a causal effect of terrain on ozone.

Future solar longitude is a known consequence of the fixed temporal sampling
interval. Computing its forecast-step values is input plumbing, not a method
contribution.

## Upload Contract

Create a standalone model file named
`models/convlstm_seasonal_topographic_evolution_operator.py`. It imports only
`torch` and exports `MODEL_SPEC`, `build_model(config)`, and all model classes
needed by the builder.

The input channel order is fixed:

```text
0: ozone
1: U wind
2: V wind
3: temperature
4: downward solar flux
```

The model rejects `in_channels != 5`. Its forward interface is:

```text
forward(x, ls, topography)

x:          [B, T, 5, H, W]
ls:         [B, T]             float32 degrees
topography: [B, 1, H, W]       float32 meters
output:     [B, K, 1, H, W]
```

The primary grid is the complete `36 x 72` global grid. The implementation
also supports smaller even-width grids used by platform dry runs. Input height
must be at least 8 and width must be even and at least 8 so the largest fixed
neighborhood and the 180-degree polar longitude shift are well defined.

Recommended model parameters are:

```text
history_hidden_dim:   32
terrain_hidden_dim:   24
operator_heads:        4
evolution_blocks:      2
dropout:              0.1
```

`history_hidden_dim` must be divisible by `operator_heads`. Neighborhood radii
are fixed at `{1, 2, 4}` to limit the experimental search space.

## Architecture Overview

The model has five stages:

```text
fixed-semantics history
    -> ozone/forcing stems
    -> full-resolution history encoder
    -> shared recurrent STEO evolution
    -> residual ozone outputs

MOLA
    -> deterministic spherical terrain descriptors
    -> shared terrain edge encoder
    -> recurrent STEO evolution

historical Ls
    -> deterministic future-step Ls
    -> harmonic seasonal encoder
    -> recurrent STEO evolution
```

The history encoder produces the initial forecast state. A shared recurrent
cell then advances that state once per forecast lead. Spatial communication in
the future-evolution core is permitted only through STEO. Pointwise dynamics
may change channels locally but may not use a spatial kernel larger than
`1 x 1`.

## Fixed-Semantics History Encoder

Separate the predicted state from its environmental forcing:

```text
O_t = x[:, t, 0:1]
F_t = x[:, t, 1:5]

Eo_t = OzoneStem(O_t)
Ef_t = ForcingStem(F_t)
E_t  = Fuse(Eo_t, Ef_t)
```

The stems and fusion projection retain full spatial resolution. A ConvLSTM
reads the fused history and returns:

```text
H_0:      [B, C, H, W]
C_hist:   [B, C, H, W]
```

`H_0` is the initial recurrent forecast state. `C_hist` is a history context
used by the pointwise future dynamics. Both summarize all five historical
channels; the model does not invent or zero-fill unknown future meteorology.

Spatial convolutions in the history encoder use the same global-grid boundary
semantics as STEO: longitude is periodic and latitude uses polar reflection.
The future evolution core itself contains no ordinary spatial convolution.

## Global-Grid Topology

For a global cell-centered `[H, W]` grid, construct latitude and angular
spacing internally:

```text
phi_i   = pi/2 - (i + 0.5) * pi/H
delta_p = pi/H
delta_l = 2*pi/W
R_mars  = 3.3895e6 meters
```

East-west distance includes the latitude metric:

```text
dx_i = R_mars * cos(phi_i) * delta_l
dy   = R_mars * delta_p
```

East-west neighbor access wraps around longitude. North-south access reflects
at a pole and shifts longitude by `W/2`, representing the 180-degree change in
longitude when crossing a pole. It never connects the northern boundary
directly to the southern boundary.

The topology helper is shared by terrain finite differences, spherical
history convolutions, and STEO neighbor gathering.

## Deterministic Terrain Geometry

Do not pass raw MOLA directly to a generic CNN. Derive a deterministic terrain
descriptor bank before any learned projection:

```text
z_norm        = z / 10000
grade_east    = (z_east - z_west) / (2*dx)
grade_north   = (z_north - z_south) / (2*dy)
slope_east    = atan(grade_east) / (pi/2)
slope_north   = atan(grade_north) / (pi/2)
slope_mag     = sqrt(slope_east^2 + slope_north^2) / sqrt(2)
aspect_east   = grade_east  / (sqrt(grade_east^2 + grade_north^2) + epsilon)
aspect_north  = grade_north / (sqrt(grade_east^2 + grade_north^2) + epsilon)
curvature     = z_norm - mean(cardinal radius-1 neighbors of z_norm)
relief_r      = (z - local_mean_r(z)) / 10000
roughness_r   = local_std_r(z) / 10000
```

Use relief radii `{1, 2, 4}` and roughness radii `{2, 4}`. The complete bank is:

```text
[z_norm,
 slope_east, slope_north, slope_mag,
 aspect_east, aspect_north,
 curvature,
 relief_1, relief_2, relief_4,
 roughness_2, roughness_4]
```

Flat terrain must produce finite zero slope, aspect, curvature, relief, and
roughness values. Physical scaling is fixed rather than learned or normalized
per sample. A shared pointwise projection maps this bank into the configured
terrain hidden dimension.

No trainable per-pixel embedding, flattened global terrain MLP, or absolute
longitude embedding is allowed. Latitude may appear only through spherical
geometry and distance factors required by the grid metric.

## Multiscale Terrain Edges

For every cell `p`, gather eight compass-direction neighbors at each radius in
`{1, 2, 4}`. For each directed edge `p -> q`, build relational features from:

```text
P_p
P_q
P_q - P_p
signed elevation difference
slope/aspect alignment with the neighbor direction
curvature difference
relief and roughness differences
normalized spherical neighbor distance
radius and direction encoding
```

The terrain edge encoder is shared across every cell, batch item, forecast
lead, and global location. Directed features allow uphill and downhill edges,
or opposite compass directions, to receive different weights while preserving
global parameter sharing.

## Deterministic Future Solar Longitude

Require `window >= 2`. Convert historical Ls to radians and compute wrapped
increments with:

```text
d_t = atan2(sin(theta_t - theta_(t-1)),
            cos(theta_t - theta_(t-1)))
```

Use at most the last four wrapped increments. Their circular mean is
`atan2(mean(sin(d_t)), mean(cos(d_t)))` and defines the local angular step.
Construct each forecast-step phase with `k=1` denoting the first future frame:

```text
theta_k = wrap(theta_last + k * mean(d_t))
```

This avoids a discontinuity at 360/0 degrees. The forecast interval spans
about 2.5 sols in the primary 20-step experiment, so local phase continuation
is the model-only fallback. If the platform later provides exact future Ls,
those values replace this calculation without changing STEO.

Encode every `theta_k` with four harmonic orders:

```text
[sin(theta_k), cos(theta_k), ...,
 sin(4*theta_k), cos(4*theta_k)]
```

## Non-Separable Seasonal-Terrain Coupling

Use a low-rank bilinear interaction rather than adding independent season and
terrain embeddings. For terrain edge embedding `E_pq` and seasonal embedding
`S_k`:

```text
U_pq     = A_e(E_pq)
V_k      = A_s(S_k)
J_pq(k)  = U_pq * V_k
logit_pq = W_j(J_pq(k)) + direction_scale_bias
```

`*` denotes elementwise multiplication. The output contains one logit per
operator head. This parameterization makes the terrain response explicitly
season dependent and provides a direct separable-additive ablation.

Normalize logits across the 24 neighbors independently for every cell, head,
batch item, and forecast step:

```text
a_pq^h(k) = softmax_q(logit_pq^h(k))
```

No small initial gate-strength parameter is used. The terrain-season branch
receives full-scale gradients from the first optimization step.

## STEO Spatial Evolution

Split normalized state channels across operator heads. For head `h`:

```text
M_p^h(k) = sum_q a_pq^h(k) * W_h(H_q - H_p)
```

Using state differences ensures that a spatially constant state yields no
terrain-driven neighbor update. Merge all heads with a pointwise projection:

```text
M_k = Merge(M_k^1, ..., M_k^heads)
```

The recurrent evolution block is:

```text
X_k = GroupNorm(H_k)
M_k = STEO(X_k, terrain_edges, season_k)
C_k = PointwiseDynamics(X_k, C_hist, season_k)

H_(k+1) = H_k + StepProjection(M_k + C_k)
```

`PointwiseDynamics` and `StepProjection` use only pointwise operations,
channel normalization, activation, and optional dropout. They cannot access a
spatial neighbor. Two evolution blocks may be applied per forecast step, with
shared parameters across forecast leads and separate parameters across the two
blocks.

Neighbor softmax, GroupNorm, and residual state updates provide architectural
stability without suppressing STEO gradients with a `0.05` branch multiplier.

## Forecast Outputs

Decode every evolved state as a residual relative to the last observed ozone
field:

```text
delta_o3_k = OutputHead(H_k)
o3_hat_k   = x[:, -1, 0:1] + delta_o3_k
```

`OutputHead` uses pointwise channel projections only; it does not introduce a
second spatial communication path after STEO.

Each forecast lead uses the last observed ozone as its reference. Do not
accumulate decoded ozone predictions from one lead to the next. The latent
state is recurrent, but direct output accumulation would unnecessarily amplify
pixel-level prediction error.

Do not apply a positivity transform because the platform may supply normalized
targets with valid negative standardized values.

## Validation And Error Handling

Reject invalid values before encoding:

- `x` must be floating point, finite, rank five, and have exactly five channels.
- Runtime window must match the configured window and be at least two.
- `ls` must be floating point, finite, on the same device as `x`, and have
  shape `[B, T]`.
- `topography` must be float32, finite, on the same device as `x`, and have
  shape `[B, 1, H, W]`.
- `x`, model parameters, and topography must use compatible dtypes; reject an
  unsupported combination with a clear `ValueError` before convolution.
- Height must be at least 8. Width must be even and at least 8.
- Positive integer configuration fields must reject booleans and zero.
- Dropout must be in `[0, 1)`.
- `history_hidden_dim` must be divisible by `operator_heads`.

The implementation may cache coordinate-only tensors by shape, device, and
dtype. It must not cache a supplied topography tensor or assume two batches
contain identical values.

## Automated Tests

Create a focused test module covering:

- upload contract, fixed channel order, schema, builder validation, and
  upload-safe imports and calls;
- standard `20 -> 20` global-grid output, platform dry-run output, finite
  backward gradients, and deterministic evaluation-mode output;
- exact longitude wrap and pole-reflection-plus-half-turn neighbor mappings;
- latitude-dependent east-west physical distance;
- flat, eastward ramp, northward ramp, and bowl terrain descriptor fixtures;
- finite flat-terrain aspect and curvature behavior;
- exact harmonic encoding and 360/0-degree future-Ls continuation;
- 24-neighbor edge construction and per-head neighbor normalization;
- spatially constant latent states producing a zero STEO difference message;
- a controlled fixture proving that one terrain edge changes differently under
  two seasonal phases;
- nonzero finite gradients for terrain, season, bilinear interaction, history,
  pointwise dynamics, and decoder parameters;
- structural verification that future pointwise dynamics contain no spatial
  convolution kernel larger than `1 x 1`;
- absence of trainable per-pixel parameters;
- primary parameter count and measured forward runtime reporting; and
- the full existing uploaded-model regression suite.

## Experimental Matrix

Train at least three matched random seeds for:

```text
B0  Persistence forecast
B1  ConvLSTM-SimVP
B2  Current Ls+MOLA gate with initial strength 1
B3  Fixed spherical neighbor operator without Ls or terrain
B4  Terrain-only dynamic operator
B5  Ls-only dynamic operator
B6  Separable additive Ls plus terrain operator
B7  Full bilinear STEO
B8  Parameter-matched ordinary spatial-convolution recurrent model
```

Keep data splits, normalization, batches, epochs, optimizer settings, and
checkpoint selection identical. Report mean and standard deviation.

Evaluate overall RMSE and stratify it by:

- forecast lead 1 through 20;
- 30-degree Ls bins;
- MOLA elevation quartiles;
- terrain slope bins;
- negative, near-zero, and positive curvature regions; and
- polar, mid-latitude, and low-latitude bands.

The full model must not be considered successful when its gain is isolated to
only short leads, one seasonal bin, or one terrain class.

## Counterfactual Diagnostics

Run inference diagnostics with:

```text
Flat:       all elevations set to zero
Rotate:     MOLA shifted by 30, 60, and 90 degrees longitude
Shuffle:    MOLA shuffled in spatial blocks
Smooth:     high-frequency terrain removed
Scale:      MOLA multiplied by 0.5 and 1.5
Ls shift:   seasonal phase shifted by selected angles
Ls fixed:   seasonal phase held constant across forecast leads
```

Counterfactual inputs do not have counterfactual ground truth. Use them to
measure prediction sensitivity and operator response, not to report causal
counterfactual accuracy.

Record per-head neighbor entropy, direction and radius weight summaries,
weight curves over Ls, and summaries stratified by terrain class. Diagnose
operator collapse when heads become indistinguishable, weights remain uniform,
or weights do not vary with both season and terrain.

## Predeclared Success Criteria

The method succeeds only if:

1. Full STEO improves mean validation RMSE over the current strength-1 joint
   gate across at least three matched random seeds.
2. Bilinear STEO improves over the parameter-matched separable additive model.
3. Improvements persist across multiple forecast leads, seasons, and terrain
   strata.
4. Both Ls and terrain counterfactuals produce measurable but numerically
   stable changes in operator behavior.
5. Operator heads exhibit differences in scale, direction, season, or terrain
   response rather than complete collapse.
6. Parameter count and runtime are reported even though they are not primary
   optimization objectives.

## Risks And Mitigations

- **Recurrent error accumulation:** supervise every lead and decode each lead
  relative to the last observation rather than the previous prediction.
- **Terrain as location lookup:** prohibit per-pixel parameters and global
  flattened terrain networks; use shared relational edges and counterfactual
  terrain diagnostics.
- **STEO bypass:** restrict all future spatial communication to STEO and keep
  the other future branch pointwise.
- **Ordinary convolution degeneration:** compare against fixed, terrain-only,
  Ls-only, separable, and parameter-matched convolution operators.
- **Head collapse:** inspect head similarity and entropy, and reduce head count
  if repeated trials show redundant heads.
- **Numerical instability:** normalize neighbor weights, normalize recurrent
  states, use residual updates, and validate all finite values.
- **Excessive physical interpretation:** describe learned weights as
  season-terrain associations unless supported by independent physical
  analysis; do not assign physical meanings to heads solely from visual form.

## Out Of Scope

- Changing the platform training loop, loss, or dataset pipeline.
- Adding future meteorological fields or a new auxiliary-input contract.
- Claiming causal terrain effects from a single fixed global MOLA map.
- Treating deterministic future Ls construction as a learned contribution.
- Adding an unrestricted ordinary spatial path parallel to STEO.
- Enforcing ozone positivity in normalized model space.
- Optimizing the first implementation for deployment latency.
