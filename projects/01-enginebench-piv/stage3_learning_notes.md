# Stage 3 - EngineBench PIV: POD/PCA

Dataset: cad090, all 1041 cycles from `LSP_r1300_p40_small.h5`.

Computed result:
- 265 of 1041 possible modes needed for 90% energy
- 717 modes needed for 99% energy
- With only 5 modes, relative reconstruction error is still 68.3% by the energy-norm metric, but the large-scale vortex/tumble structure is already visually recognizable
- With 265 modes (90% energy), relative error drops to 23.9%; with 717 (99%), 7.6%

Figures: `results/stage3_pod_energy.png`, `results/stage3_pod_reconstruction.png`

## Physical observations (your own words)

1. It's a bit counterintuitive that I need 265 modes to hit 90% energy but only 5 modes to see the vortex clearly. Most of the "energy" here is fine turbulent fluctuation, not the big coherent structure, even though the big structure is what visually stands out.
2. The 5-mode reconstruction actually looks pretty close to the original for the large features - it's really only the small-scale texture that's missing. If I only cared about the tumble/vortex shape, 5 modes would basically be enough.
3. This makes me think POD energy percentage is a bit of a misleading metric on its own if the goal is capturing the physically important structure rather than raw variance.

## ML/data concepts - what I now understand

- Eigenvectors/eigenvalues in this context (what do POD modes and singular values physically represent here?):
- Latent representation (how is the POD coefficient vector a compressed representation of the flow?):
- Compression vs. prediction (why is 90%+ reconstruction fidelity not the same thing as being able to predict a NEW field?):

## What I'm still unsure about

-
