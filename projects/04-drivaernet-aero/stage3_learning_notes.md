# Stage 3 - SHAP feature importance for Cd (tabular model)

Top features by SHAP importance: fender/wheel-arch offset, diffuser angle, A/B/C-pillar thickness, trunklid length, rear window length.

## Physical observations (my own words)

1. The top 3 SHAP features exactly match Stage 1's top 3 raw single-parameter correlations - that agreement is reassuring, but it's also expected here, not a strong independent validation, since with a weak overall model (R2=0.563) there isn't much room for SHAP to reveal something a simple correlation wouldn't already show.
2. The directions make real aerodynamic sense: bigger fender/wheel-arch offset and steeper diffuser angle both increase Cd (matches known underbody/wheel-well flow separation physics), while thicker A/B/C pillars DECREASE Cd - which is less obviously intuitive to me, but plausible if it correlates with a smoother, less abrupt greenhouse transition in this parameterization rather than pure frontal-area blockage.
3. Given Stage 2's honest finding that this tabular representation only explains ~56% of Cd variance, I'm treating this SHAP ranking as "the story the summary parameters can tell," not "the complete story of what drives drag on these cars." The other ~44% is exactly the kind of thing genuine 3D shape data (the sketches, if I get them) might explain that 23 numbers can't.

## What I'm still unsure about

- Why pillar thickness has a strong NEGATIVE effect on Cd - I have a plausible guess (smoother greenhouse transition) but haven't confirmed it against the actual geometry, which is exactly the gap real shape data would close.
