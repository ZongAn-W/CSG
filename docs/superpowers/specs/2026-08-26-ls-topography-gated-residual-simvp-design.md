# Ls-Topography Gated Residual SimVP Design

## Goal

Extend `convlstm_ls_topography_joint_gated_simvp.py` so solar longitude and
topography can both modulate existing encoded dynamics and add new conditional
features. Preserve the current AresVision upload contract and the unchanged
ConvLSTM-SimVP main path.

## Upload Contract

The model continues to accept `forward(x, ls, topography)` where `x` has shape
`[B,T,C,H,W]`, `ls` has shape `[B,T]`, and static MOLA topography has shape
`[B,1,H,W]`. It returns `[B,horizon,1,H,W]`. AresVision does not provide future
Ls values, so this change uses historical Ls only.

The schema adds `initial_residual_strength`, a float in `[0,1]` with default
`0.05`. Existing parameters and auxiliary metadata remain unchanged.

## Conditional Fusion

The existing gate constructs a shared hidden representation from projected
encoded features, harmonic Ls features, and encoded topography:

```text
joint = GELU(feature_projection(encoded) + ls_features + terrain_features)
gate = tanh(gate_projection(joint))
residual = residual_projection(joint)
output = encoded * (1 + gate_strength * gate)
       + residual_strength * residual
```

Gate and residual output projections use small-gain Xavier initialization and
zero bias. This keeps the initial model close to the current implementation
while giving both branches nonzero gradients on the first backward pass.
Both trainable strengths use the existing straight-through bounded `[0,1]`
parameterization.

The additive residual has the same `[B,T,S,h,w]` shape as the encoded history.
It enters before the unchanged temporal translator, allowing the translator to
use newly supplied seasonal and terrain features for every forecast lead.

## Compatibility

With `initial_residual_strength=0`, shared legacy weights produce output exactly
equal to the current gated model. With both strengths zero, the model continues
to reduce exactly to `convlstm_simvp.py`. Main-path class ASTs remain identical
to the baseline.

## Tests

Update the dedicated test module before implementation. Cover schema and
configuration validation, residual tensor shape, finite gradients for every
branch, exact zero-residual compatibility with the legacy equation, additive
signal when encoded features are zero, standard and odd input shapes, parameter
overhead, upload-safe imports, and the complete existing regression suite.
