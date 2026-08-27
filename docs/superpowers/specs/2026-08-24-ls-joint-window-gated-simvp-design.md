# Ls-Conditioned Joint Window-Gated SimVP Design

## Goal

Create an independent AresVision upload model that tests whether jointly
computed, Ls-conditioned input-window scaling improves the C-S-1 baseline,
especially for a forecast horizon of 20.

## Experimental Boundary

The model preserves the ConvLSTM encoder, spatial encoder, SimVP temporal
translator, and spatial decoder from `models/convlstm_simvp.py`. The only
structural change is one window gate inserted after spatial encoding and before
the temporal translator. The model does not use attention, softmax, historical
weighted sums, a state-motion stream, last-state replacement, an additional
forecast head, or an additional loss.

## Upload Contract

Create `models/convlstm_ls_joint_window_gated_simvp.py` with only `torch`
imports. It exports `MODEL_SPEC`, `build_model(config)`, and a model whose
forward method accepts `x` with shape `[B,T,C,H,W]` and `ls` with shape `[B,T]`.
The output has shape `[B,horizon,1,H,W]`. Ls is required floating-point solar
longitude in degrees, finite, and on the same device as `x`.

The schema retains every C-S-1 parameter and adds:

- `window_gate_hidden_dim`: integer, default 32, minimum 4, maximum 256.
- `initial_window_strength`: float, default 0.05, minimum 0.0, maximum 1.0.

## Joint Window Gate

Spatial encoding produces `encoded` with shape `[B,T,S,h,w]`. Global average
pooling produces per-window descriptors `[B,T,S]`. For every input step, the
gate constructs first- and second-order Ls harmonics in this order:
`sin(Ls)`, `cos(Ls)`, `sin(2Ls)`, `cos(2Ls)`. Concatenating descriptors and
harmonics gives `[B,T,S+4]`.

The full tensor is flattened to `[B,T*(S+4)]` and processed by:

```text
Linear(T*(S+4), window_gate_hidden_dim)
GELU
Linear(window_gate_hidden_dim, T)
Sigmoid
```

Flattening before the MLP ensures that every output weight can depend on every
historical feature and every historical Ls value. Sigmoid decisions remain
independent and are not normalized across time.

The output layer uses small-gain Xavier initialization and zero bias, so initial
weights are close to 0.5. A bounded trainable strength starts at the configured
value. The gate applies:

```text
scale = 1 + strength * (2 * weight - 1)
weighted = scale[:, :, None, None, None] * encoded
```

With the default strength, every initial scale lies in `(0.95, 1.05)`, keeping
the initial path close to C-S-1 while allowing gate gradients on the first
backward pass. Every input window, including the final window, remains
controllable.

## Data Flow

1. Validate `x` and `ls`.
2. Encode the history with the unchanged C-S-1 ConvLSTM.
3. Apply the unchanged shared spatial encoder.
4. Compute joint Ls-conditioned window weights and residual scales.
5. Flatten the scaled history to `[B,T*S,h,w]`.
6. Apply the unchanged C-S-1 temporal translator.
7. Apply the unchanged C-S-1 spatial decoder.

The shared `[B,T]` weights are used for all forecast leads. This experiment
tests history selection only; horizon-specific history selection is outside its
scope.

## Validation And Tests

Create `tests/test_convlstm_ls_joint_window_gated_simvp.py`. First verify the
required missing-model-file red failure. Tests cover the upload schema and AST
safety, standard and horizon-20 shapes, multichannel and odd spatial sizes,
`window=1`, harmonic correctness, bounded non-normalized weights, joint
cross-window dependence, last-window controllability, near-identity scaling,
finite nonzero gate gradients, finite gradients for every trainable parameter,
invalid Ls inputs, and AST equivalence of all C-S-1 main-path classes. Run the
new test file and the complete existing suite.
