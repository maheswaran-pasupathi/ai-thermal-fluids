# Stage 1 - EngineBench PIV: learning notes

Dataset: EngineBench LSP Small (`LSP_r1300_p40_small.h5`, downloaded via Kaggle API)
Test point: r1300_p40 (1300 RPM, 40 kPa MAP) | Crank angles available: cad090, cad135, cad180, cad225, cad270 | 1041 cycles per crank angle
Figures: `results/stage1_piv_snapshot.png` (U/V/magnitude/quiver/vorticity at cad090), `results/stage1_crank_angle_comparison.png` (magnitude across 4 crank angles)

## Physical observations (your own words, 3 minimum)

1. The flow isn't uniform at all - there's a large swirling structure taking up almost half the field, with fast negative U on the left and fast positive U on the right. That's basically the tumble motion you'd expect in-cylinder, not a clean uniform flow.
2. Velocity magnitude is patchy rather than smooth - pockets of high speed sitting right next to low-speed regions. That's turbulence riding on top of the mean tumble motion, not something a laminar flow would look like.
3. The vorticity field is noisy and doesn't show one clean vortex core - it's scattered, especially near the edges. Probably a mix of real small-scale turbulence and PIV measurement noise near the boundary of the field of view.

## ML/data concepts - what I now understand

- Sample, feature, target (in this dataset, what is each?):
- Arrays/tensors and dimensions (what does the (n_snapshots, n_points, 4) shape mean?):
- Normalization (why the dataset stores mean_u/std_u/mean_v/std_v as attributes):
- Train/validation/test (not used yet today, but how would you split this data?):
- Spatial-field data vs ordinary tabular data (how is this different from a row-per-sample CSV?):

## What I'm still unsure about

-

## LinkedIn post draft (optional, only after real results exist)

