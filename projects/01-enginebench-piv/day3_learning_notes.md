# Day 3 - EngineBench PIV: POD/PCA

Dataset: cad090, all 1041 cycles from `LSP_r1300_p40_small.h5`.

Computed result:
- 265 of 1041 possible modes needed for 90% energy
- 717 modes needed for 99% energy
- With only 5 modes, relative reconstruction error is still 68.3% by the energy-norm metric, but the large-scale vortex/tumble structure is already visually recognizable
- With 265 modes (90% energy), relative error drops to 23.9%; with 717 (99%), 7.6%

Figures: `results/day3_pod_energy.png`, `results/day3_pod_reconstruction.png`

## Physical observations (your own words)

1. (Why does the large-scale structure show up with just 5 modes, while the %-energy metric says you need hundreds more?)
2.
3.

## ML/data concepts - what I now understand

- Eigenvectors/eigenvalues in this context (what do POD modes and singular values physically represent here?):
- Latent representation (how is the POD coefficient vector a compressed representation of the flow?):
- Compression vs. prediction (why is 90%+ reconstruction fidelity not the same thing as being able to predict a NEW field?):

## What I'm still unsure about

-
