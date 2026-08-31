# Stage 2 - thermal KPI surrogate: predicting max rack-back temperature

Combined 1428 rows (previous + retrofitted design sensor logs) with 5 cooling-unit operational features (airflow, cool output, fan speed, return/supply air temp) predicting max rack-back temperature.

Results: Linear MAE=0.30C, R2=0.493. Random Forest MAE=0.06C, R2=0.952.

## Physical observations (my own words)

1. Random Forest's R2=0.952 looks great on its own, but I need to be honest about what it actually means here: the target only spans 33.1-34.5C across the whole dataset (std=0.48C). This isn't a designed sweep across genuinely different cooling regimes - it's ~2 hours of near-steady operation for each design, logged every 10 seconds. A high R2 on a narrow, slowly-varying window is a much weaker claim than a high R2 across a genuinely diverse operating envelope.
2. That said, the surrogate did learn something real, not just autocorrelation - the linear baseline (which can't exploit short-term temporal smoothness the way a nearest-neighbor-like Random Forest split can) still gets R2=0.493, which tells me there IS a real relationship between cooling settings and rack temperature here, just a weak one within this narrow window.
3. For Stage 3's optimization to be honest, I need to stay within the operating range this surrogate was actually trained on (33-34.5C, current fan-speed/airflow range) rather than claim it can find a better setting far outside anything it's seen.

## ML/data concepts - what I now understand

- High R2 alone doesn't tell you whether a model learned real structure or just exploited a narrow, smooth, autocorrelated target range - I need to check the target's actual variance/range before trusting the number.
- `merge_asof` was the right tool for joining two independently-logged time series (cooling metrics, rack sensors) that don't share exact timestamps - an exact join would have silently dropped most rows.
- Pooling both design variants into one training set is defensible here (same physical system, same sensor definitions) but I'm treating "design" as an unlabeled condition rather than checking whether the surrogate secretly relies on which design it's looking at - worth a follow-up check.

## What I'm still unsure about

-
