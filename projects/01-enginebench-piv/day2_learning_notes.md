# Day 2 - EngineBench PIV: cycle-to-cycle variability (EDA)

Dataset: same file as Day 1 (`LSP_r1300_p40_small.h5`), all 5 crank angles, 1041 cycles each.

Computed result: field-averaged cycle-to-cycle std of velocity magnitude by crank angle:
- cad090: 5.357
- cad135: 4.106
- cad180: 2.485
- cad225: 1.984
- cad270: 1.829

Figure: `results/day2_cycle_variability.png`

## Physical observations (your own words)

1. (Why might variability be highest early in the cycle at cad090 and fall off toward cad270?)
2.
3.

## ML/data concepts - what I now understand

- EDA (what did looking at mean/std across cycles tell you that a single snapshot couldn't?):
- Variance / noise vs. real signal (how do you tell cycle-to-cycle variability apart from measurement noise?):
- Train/val/test splitting without leakage (if you split these 1041 cycles for later ML, what would go wrong if you split randomly instead of by cycle-block?):

## What I'm still unsure about

-
