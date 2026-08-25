# Ls-Topography Joint-Gated ConvLSTM-SimVP Design

## Goal

Create an independent AresVision uploaded model that tests whether a small,
joint solar-longitude and MOLA topography gate improves validation ozone RMSE
over `models/convlstm_simvp.py`. The primary experiment uses 20 historical
frames to predict 20 future ozone frames. Input channels include historical
ozone, U wind, V wind, temperature, and solar radiation flux.

## Experimental Boundary

The ConvLSTM encoder, spatial encoder, SimVP temporal translator, and spatial
decoder remain structurally identical to the baseline. The only main-path
change is a residual multiplicative gate between spatial encoding and the
temporal translator. The platform training loop, loss, data split,
normalization, metrics, and checkpoint behavior do not change.

The gate conditions historical features only. It does not extrapolate future
Ls, add topography to `in_channels`, replace the baseline translator, add a
second prediction head, or require an auxiliary loss.

## Upload Contract

Create `models/convlstm_ls_topography_joint_gated_simvp.py`. The file imports
only `torch` and exports `MODEL_SPEC`, `build_model(config)`, and the model
classes used by the builder.

`MODEL_SPEC["auxiliary_inputs"]` declares both required inputs exactly:

- `ls`: float32 degrees with shape `[batch, window]`.
- `topography`: float32 meters with shape
  `[batch, 1, height, width]`.

The model method is `forward(self, x, ls, topography)`. Positional order is
always `x`, `ls`, then `topography`. Input `x` has shape
`[B,T,C,H,W]`, and the output has shape `[B,K,1,H,W]`. Static topography has no
time axis and is not included in `config["in_channels"]`.

The schema retains the five baseline parameters and adds:

- `gate_hidden_dim`: integer, default 32, minimum 4, maximum 256.
- `initial_gate_strength`: float, default 0.05, minimum 0.0, maximum 1.0.

Topography uses a fixed 10,000-meter physical scale rather than an additional
search parameter.

## Architecture

The unchanged ConvLSTM and spatial encoder produce history features:

```text
E: [B,T,S,h,w]
```

Three lightweight branches form the gate context.

### Dynamic Feature Branch

A shared pointwise projection maps every encoded history feature from `S` to
`G=gate_hidden_dim` channels:

```text
W_e(E): [B,T,G,h,w]
```

### Solar-Longitude Branch

For every historical Ls value in degrees, construct first- and second-order
periodic features in this order:

```text
sin(Ls), cos(Ls), sin(2Ls), cos(2Ls)
```

Angles are converted to radians first. A shared per-step MLP maps the four
features to `G` channels. The result is reshaped to `[B,T,G,1,1]` and broadcast
over space. Periodic encoding avoids a discontinuity between 359 and 0 degrees.

### Topography Branch

Divide MOLA elevation in meters by 10,000 before encoding. This keeps typical
values numerically moderate while preserving the sign and absolute elevation
reference. Do not subtract a per-sample spatial mean.

A two-layer convolutional encoder downsamples the static terrain to the same
`[h,w]` grid as `E` and produces `P` with shape `[B,G,h,w]`. If an odd input
size causes a mismatch, interpolate `P` to the exact encoded spatial size.
Broadcast `P` over the history dimension without materializing copies.

### Joint Residual Gate

Fuse all three branches before the nonlinearity:

```text
J = GELU(W_e(E) + L + P[:, None])
gate = tanh(W_g(J))
strength = bounded_trainable_strength(initial_gate_strength)
scale = 1 + strength * gate
E_gated = scale * E
```

`W_g` is a shared pointwise projection from `G` back to `S`, so `gate` has
shape `[B,T,S,h,w]`. Positive values enhance a feature and negative values
suppress it. The nonlinear fusion lets the Ls response vary with terrain and
the dynamic atmospheric state without introducing a large attention module.

Initialize the final gate projection with small-gain Xavier weights and zero
bias. With the default strength, the initial feature scale remains close to
the interval `[0.95, 1.05]`, while all conditioning branches can receive
gradients on the first backward pass. Use a straight-through bounded strength
in `[0,1]`, following the existing residual-gate pattern. A configured strength
of zero gives the exact baseline feature path:

```text
E_gated = E
```

Flatten `E_gated` to `[B,T*S,h,w]`, then run the unchanged temporal translator
and decoder to produce `[B,K,1,H,W]`.

## Validation And Errors

Reject invalid inputs before encoding:

- `x` must be a rank-five tensor with the configured window and channel count.
- `ls` must be present, floating point, finite, on the same device as `x`, and
  have shape `[B,T]`.
- `topography` must be present, floating point, finite, on the same device as
  `x`, and have shape `[B,1,H,W]`.
- Integer configuration fields must be positive, dropout must be in `[0,1)`,
  and the initial gate strength must be in `[0,1]`.

Raise clear `ValueError` messages. Do not substitute zeros, flat terrain,
random values, or `None` for missing auxiliary data. The platform owns MOLA
coordinate alignment; the model verifies the resulting tensor contract.

## Automated Verification

Add `tests/test_convlstm_ls_topography_joint_gated_simvp.py` covering:

- required exports, exact auxiliary metadata, upload-safe imports, and banned
  call checks;
- platform dry-run shape, the primary 20-to-20 case, multichannel inputs, odd
  spatial sizes, and full finite backpropagation;
- exact Ls harmonic values at 0, 90, 180, and 270 degrees;
- topography scaling that preserves signed absolute elevation;
- gate shape, finite values, initial scaling bounds, and sensitivity to both
  Ls and topography;
- non-separable joint behavior, checked by verifying that an Ls change has a
  different gate effect under two distinct terrain fields;
- exact baseline output equivalence when the shared main-path weights match
  and `initial_gate_strength` is zero;
- finite gradients for every trainable parameter and clear failures for every
  invalid Ls, topography, tensor-shape, device, and configuration branch;
- AST equivalence of the baseline ConvLSTM, spatial encoder, translator, and
  decoder classes; and
- parameter overhead below 15 percent of the baseline for the primary 20-to-20
  configuration.

Run the focused test file and the complete existing test suite.

## Evaluation Plan

Train three controlled configurations with identical data splits,
normalization, batches, epochs, optimizer settings, and random seeds:

1. Original `convlstm_simvp.py` baseline.
2. Existing Ls-only gated model.
3. New Ls-plus-MOLA joint-gated model.

The existing Ls-only model uses a different gate parameterization. Its score
provides useful context but is not a matched ablation that isolates the net
effect of MOLA. This implementation cycle may establish that the joint model
improves the baseline, but it must not attribute the full difference from the
Ls-only model to topography alone.

Use at least three random seeds and report mean and standard deviation. The
primary success criterion is lower overall validation ozone RMSE than the
baseline. Also report RMSE for each forecast lead from 1 through 20, for 30
degree Ls bins, and for MOLA elevation quartiles. Treat a gain as robust only
when it is not caused by a narrow seasonal, elevation, or short-lead subset.

Record gate mean, standard deviation, minimum, and maximum during evaluation.
Compare gate summaries across Ls bins and elevation bins to verify that the
mechanism responds to both conditions and does not remain saturated. These
associations support model diagnosis but are not causal physical claims.

## Out Of Scope

- Future-Ls extrapolation or future exogenous forcing.
- Injecting auxiliary conditions into the internal ConvLSTM gates.
- A horizon-specific post-translator gate.
- An additional topography-only uploaded model in this implementation cycle.
- Changes to the platform training or metric pipeline.
