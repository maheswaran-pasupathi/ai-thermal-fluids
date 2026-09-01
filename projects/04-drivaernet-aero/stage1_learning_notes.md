# Stage 1 - DrivAerNet++ design-space EDA

4165 real car designs, 23 geometry parameters, CFD-derived Cd/Cl coefficients.

## Physical observations (my own words)

1. Cd ranges 0.20-0.32 across the dataset - a realistic range for real road cars, not an unphysical spread. The distribution is smooth and roughly unimodal, which tells me this is a genuinely continuous design sweep, not several disconnected clusters of car types.
2. The top-correlated single parameters make real aerodynamic sense: diffuser angle and fender/wheel-arch offset are both well-known drag drivers in automotive aero (diffuser angle affects underbody flow recovery, wheel-arch geometry affects flow separation around the wheels) - not something I'd expect a random/spurious correlation to land on by chance.
3. Even the strongest single-parameter correlation (r=0.414, fender arch offset) shows a lot of scatter around the trend - Cd spans nearly 0.10 at almost every value of that one parameter. That tells me drag here is genuinely multivariate: no single geometry knob controls it, which is exactly why Stage 2 needs a model that can combine multiple parameters, not just the strongest univariate one.

## ML/data concepts - what I now understand

- Correlation coefficient magnitude and "does the scatter look tight" are different questions - r=0.414 sounds moderate, but visually the scatter is wide enough that no one would mistake this for a simple 1-parameter relationship.
- A smooth, wide, continuous target distribution (not clustered/multimodal) is a good sign for regression - it suggests the design space was sampled reasonably densely, not just a few discrete car archetypes.

## What I'm still unsure about

-
