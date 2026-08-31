# Stage 2 - EngineBench PIV: cycle-to-cycle variability (EDA)

Dataset: same file as Stage 1 (`LSP_r1300_p40_small.h5`), all 5 crank angles, 1041 cycles each.

Computed result: field-averaged cycle-to-cycle std of velocity magnitude by crank angle:
- cad090: 5.357
- cad135: 4.106
- cad180: 2.485
- cad225: 1.984
- cad270: 1.829

Figure: `results/stage2_cycle_variability.png`

## Physical observations (your own words)

1. Variability is highest right after cad090 and drops steadily as the cycle progresses - whatever's driving it early (probably intake-generated turbulence still working through the cylinder) seems to be less repeatable cycle-to-cycle than what happens later.
2. It's not just the variability that changes - the mean flow pattern itself shifts shape between crank angles, with the high-speed region moving from one side of the field toward the other as CAD increases. This isn't the same structure just decaying, it's an evolving flow.
3. By cad270 the flow looks noticeably calmer and more repeatable, which physically tracks with the tumble motion breaking down before compression ramps up.

## ML/data concepts - what I now understand

- EDA (what did looking at mean/std across cycles tell you that a single snapshot couldn't?):
- Variance / noise vs. real signal (how do you tell cycle-to-cycle variability apart from measurement noise?):
- Train/val/test splitting without leakage (if you split these 1041 cycles for later ML, what would go wrong if you split randomly instead of by cycle-block?):

## What I'm still unsure about

-
