# Stage 2 - Cd regression from 23 scalar geometry parameters

3332 train / 833 test split. Linear MAE=0.0126, R2=0.496. Random Forest MAE=0.0116, R2=0.528. XGBoost MAE=0.0115, R2=0.563.

## Physical observations (my own words)

1. This is genuinely weaker than what I got in Projects 1-3 (R2 0.76-0.95 there vs. 0.563 here), and I don't think that's a modeling failure - it's a real limitation of the input representation. 23 scalar parameters summarize the car's shape, but they can't capture surface curvature detail, local flow separation points, or interaction effects between features that aren't in the summary. A drag coefficient is the integral of pressure and shear over the ENTIRE surface - reducing that surface to 23 numbers throws away real information.
2. The scatter plot makes this concrete: at the extremes (Cd<0.22 or Cd>0.29), predictions compress toward the middle rather than tracking the actual value - classic behavior when the input features don't carry enough signal to distinguish extreme designs from typical ones.
3. MAE being 52% of Cd's own standard deviation is the honest way to say this: the model is doing real work (much better than guessing the mean every time, which would be 100%), but it's not close to design-screening-grade accuracy yet.

## Why I'm not stopping here

This result directly supports something raised mid-session: scalar parameters aren't the same as understanding the actual geometry. DrivAerNet++ also provides real per-design 2D sketches (silhouette images) - a genuinely different, richer representation of shape than 23 summary numbers. Adding those (once downloaded - gated behind a one-time Harvard Dataverse form) is a real next step, not just a nice-to-have, given how much variance the tabular model leaves unexplained.

## ML/data concepts - what I now understand

- R2 needs to be judged against what the input features could plausibly capture, not just against other projects' numbers - a weaker R2 here is expected given a genuinely lower-information input representation, not a sign I did something wrong.
- Regression-to-the-mean at the extremes (visible in the scatter plot, not just inferable from the R2 number) is a specific, diagnosable symptom of insufficient input information, distinct from generic "the model needs tuning."

## What I'm still unsure about

- Whether hyperparameter tuning would meaningfully close this gap, or whether it's fundamentally an information-content ceiling that no amount of RF/XGBoost tuning fixes without richer input features.
