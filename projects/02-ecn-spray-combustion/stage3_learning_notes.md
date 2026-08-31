# Stage 3 - SHAP feature importance vs. combustion physics

Model: the regularized XGBoost from Stage 2 (depth=3, reg_lambda=2), SHAP TreeExplainer on the test set.

Importance ranking (mean |SHAP value|, mm): Ta (7.89) > dens (7.26) > injP (2.01) > oxycon (1.66) > orifDiam (0.67).

Direction check against known combustion physics - all 5 matched my prior expectation:
- Ta: -0.947 (higher ambient temp -> shorter lift-off) - matches faster ignition kinetics
- oxycon: -0.915 (higher O2 -> shorter lift-off) - matches faster reaction/more oxidizer
- dens: -0.786 (higher ambient density -> shorter lift-off) - matches denser air, more mixing
- injP: +0.475 (higher injection pressure -> longer lift-off) - matches higher jet velocity pushing the flame downstream
- orifDiam: +0.038 (essentially no simple linear direction) - matches my expectation going in that orifice diameter's effect is more complex/non-monotonic, not a clean textbook sign

## Physical observations (my own words)

1. Every single direction matched what I expected from combustion chemistry before I even ran this - that's a real, meaningful check that the model learned actual physics rather than a spurious pattern in a pooled multi-study dataset. If even one of these had come out backwards, I'd trust the whole model a lot less regardless of its R2.
2. Ambient temperature and density dominate importance together (7.89 and 7.26, both far ahead of injP/oxycon/orifDiam) - which makes sense, since together they set the local reaction rate and mixing environment the fuel jet actually ignites into, while injection pressure and O2 are more secondary tuning parameters on top of that.
3. Orifice diameter being both the least important AND having no clean direction isn't a weakness in the analysis - it's consistent with Stage 2's finding that this exact feature is where the model's generalization actually breaks down (the 0.894mm orifice). A feature the model can't reason about cleanly is also the one it can't extrapolate on cleanly.

## ML/data concepts - what I now understand

- SHAP explains a trained model's actual behavior, not just correlations in the raw data - it's decomposing individual predictions, not restating an EDA scatter plot with extra steps.
- A feature-importance ranking alone (the bar chart) is less useful than the direction-of-effect view (the beeswarm) - "important" and "physically sensible" are two different questions, and I need both answered before trusting the model.
- Checking model explanations against domain knowledge is itself a validation step, not just a nice-to-have visualization - this is the actual reason to reach for SHAP over just reporting R2.

## What I'm still unsure about

-
