# Day 4 - EngineBench PIV: crank angle -> POD coefficients -> mean field regression

Setup: global POD basis (50 modes) pooled across all 5 crank angles. Linear regression: crank angle (degrees) -> POD coefficients, trained on cad090/135/225/270, evaluated on held-out cad180 (interpolation, not extrapolation).

Result:
- Global linear regression (all 4 training crank angles) relative error vs actual mean field: **77.4%**
- Naive baseline (just reuse nearest known crank angle, cad135): **78.5%**
- Improvement tested: local interpolation between just the two nearest known crank angles (cad135, cad225) instead of a global fit: **55.2%** - a real, meaningful improvement.

The global regression barely beat the naive baseline - a real negative/limited result, reported honestly rather than dropped. Testing local interpolation as a fix confirmed the diagnosis: the bottleneck was trying to fit one trend across 4 widely-spaced points, not the general approach. See `results/day4_regression_result.png` for all three compared side by side.

## Physical observations (your own words)

1. (Why would in-cylinder flow structure NOT vary linearly with crank angle between 90 and 270 degrees?)
2.
3.

## ML/data concepts - what I now understand

- Overfitting/generalization with tiny sample sizes (why is fitting a model to 4 data points not meaningful evidence of a working model?):
- Interpolation vs extrapolation (why was cad180 chosen as the held-out point instead of cad090 or cad270?):
- Baseline comparison (why does comparing against a naive baseline matter more than the raw error number alone?):

## What would change with the full dataset

The full EngineBench dataset (not just this 5-crank-angle small subset) has many more crank angles and operating points. More densely sampled conditions would let a regression model actually learn the nonlinear crank-angle -> flow relationship instead of interpolating between 2 widely-spaced points. Logged as Phase 2 backlog in the project README rather than attempted here.

## What I'm still unsure about

-
