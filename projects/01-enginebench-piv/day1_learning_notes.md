# Day 1 - EngineBench PIV: learning notes

Dataset: EngineBench LSP Small (`LSP_r1300_p40_small.h5`, downloaded via Kaggle API)
Test point: r1300_p40 (1300 RPM, 40 kPa MAP) | Crank angles available: cad090, cad135, cad180, cad225, cad270 | 1041 cycles per crank angle
Figures: `results/day1_piv_snapshot.png` (U/V/magnitude/quiver/vorticity at cad090), `results/day1_crank_angle_comparison.png` (magnitude across 4 crank angles)

## Physical observations (your own words, 3 minimum)

1.
2.
3.

## ML/data concepts - what I now understand

- Sample, feature, target (in this dataset, what is each?):
- Arrays/tensors and dimensions (what does the (n_snapshots, n_points, 4) shape mean?):
- Normalization (why the dataset stores mean_u/std_u/mean_v/std_v as attributes):
- Train/validation/test (not used yet today, but how would you split this data?):
- Spatial-field data vs ordinary tabular data (how is this different from a row-per-sample CSV?):

## What I'm still unsure about

-

## LinkedIn post draft (optional, only after real results exist)

