# Day 5 - EngineBench PIV: sparse/gappy-POD reconstruction

Technique: gappy POD (Everson & Sirovich, 1995) - fit POD coefficients by least squares using only the known (unmasked) points, then use those coefficients to reconstruct the full field including the masked gaps.

Setup: cad090, POD basis from 1040 training snapshots (100 modes), tested on 1 held-out snapshot with 30% of spatial points randomly masked.

Result:
- Relative error, gap points only: **46.3%**
- Large-scale vortex/tumble structure recovers well visually even with 30% missing; fine-scale turbulent detail does not - consistent with the Day 3 finding that few modes capture large structure but many are needed for fine-scale energy.

Figure: `results/day5_sparse_reconstruction.png`

## Physical observations (your own words)

1.
2.
3.

## ML/data concepts - what I now understand

- Least-squares fitting with a reduced basis (why can 100 POD modes be fit from only 70% of the spatial points?):
- What "gap fraction" and "reconstruction fidelity" trade off against each other:
- How this connects to real PIV experiments (when would sparse/incomplete PIV actually happen in practice?):

## What I'm still unsure about

-
