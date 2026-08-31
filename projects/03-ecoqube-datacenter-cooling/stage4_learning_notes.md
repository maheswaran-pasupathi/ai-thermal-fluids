# Stage 4 - dashboard interface (static, not Streamlit)

Wrapped Stages 2-3 into two functions: `predict_hotspot(settings)` and `recommend_setting(safe_limit)`, plus an explicit out-of-range flag so a caller can tell a real prediction apart from an unsupported extrapolation.

## Physical observations (my own words)

1. The out-of-range check actually caught something in my own demo: pushing fan speed to 150% (physically meaningless anyway, fan speed is a percentage) still returned a prediction from the Random Forest, because tree models don't refuse to extrapolate the way a person would - they just clip to the nearest leaf. Without the explicit range check, that silently-wrong prediction would look identical to a trustworthy one.
2. I chose static functions over a Streamlit app given the time budget - the curriculum's own plan allows either, and a static interface with an explicit range-safety check is more honest about what this actually is (a portfolio demo, not a deployed tool) than a polished-looking dashboard would be.

## What I'm still unsure about

- A real deployment would need actual bounds/alerting on the "recommend_setting" search too, not just the prediction function - right now the optimizer in Stage 3 silently trusts the grid it's given.
