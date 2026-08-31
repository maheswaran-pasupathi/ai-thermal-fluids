# Stage 2 - lift-off length regression: linear vs Random Forest vs XGBoost

Dataset: 336 clean rows from Stage 1, 75/25 train/test split (252/84).

Results (test set):
- Linear: MAE=7.63mm, RMSE=11.21mm, R2=0.550
- Random Forest: MAE=5.15mm, RMSE=8.17mm, R2=0.761
- XGBoost: MAE=4.57mm, RMSE=7.54mm, R2=0.797

Overfitting check (train R2 vs test R2):
- Linear: 0.616 -> 0.550 (gap 0.066)
- Random Forest: 0.975 -> 0.761 (gap 0.213)
- XGBoost: 0.992 -> 0.797 (gap 0.195)

## Physical observations (my own words)

1. The linear model actually predicts a negative lift-off length for one test point, which is physically impossible - lift-off length can't be less than zero. Neither RF nor XGBoost ever does that, since tree-based models can't extrapolate outside the range of values they saw in training the way a straight line can.
2. XGBoost beating Random Forest by a meaningful margin (R2 0.797 vs 0.761) with a fairly shallow tree depth (4) suggests there's real structure in this data that boosting captures better than plain bagging - not a huge gap, but consistent with what I'd expect from these two algorithms on tabular engineering data.
3. Both tree models show a real train-test gap (~0.20), exactly matching the clustering concern I flagged in Stage 1 - injection pressure and orifice diameter cluster around a handful of common values, so the model is partly memorizing those clusters rather than learning a fully generalizable relationship. That's not a bug, it's an honest limitation of a dataset pooled from multiple studies rather than one designed sweep.

## ML/data concepts - what I now understand

- A model that predicts an impossible value (negative length) even once is a real red flag about extrapolation behavior, not just a minor accuracy issue - worth checking even when the aggregate R2 looks fine.
- Train R2 vs test R2 gap is the actual overfitting signal, not test R2 alone - both tree models look "good" on test R2 but the size of that gap tells me how much to trust it holding up on genuinely new conditions.
- Boosting (XGBoost) and bagging (Random Forest) reach different bias/variance tradeoffs even on the same shallow-tree budget - not automatic that boosting wins, but it did here.

## Follow-up: is the train-test gap a data-parsing bug or a real generalization limit?

Checked properly rather than assumed: no duplicate rows, and the 33 rows flagged as statistical outliers are all legitimate distinct experimental sub-series (different orifice/pressure regimes pooled from different studies), not corrupted values - confirmed by inspection, not just a summary stat.

That pointed at the real cause: random k-fold CV doesn't respect the fact that this table pools genuinely different rigs/nozzle hardware. I tested this directly with a leave-one-orifice-size-out analysis - train on every OTHER orifice diameter, test only on the held-out one:

| Held-out orifice (mm) | n | R2 | MAE (mm) |
|---|---|---|---|
| 0.084 | 8 | 0.848 | 2.61 |
| 0.091 | 24 | 0.719 | 4.90 |
| 0.1 | 26 | 0.881 | 3.97 |
| 0.18 | 210 | 0.694 | 5.73 |
| 0.246 | 21 | 0.923 | 3.25 |
| 0.363 | 10 | 0.891 | 3.86 |
| 0.894 | 34 | **0.354** | **9.25** |

Six of seven orifice sizes generalize well (R2 0.69-0.92) even when completely held out - real evidence the model learned transferable physics, not just memorized clusters. The 0.894mm orifice is the one clear failure, and it makes physical sense: it's 5-10x larger than the 0.09-0.18mm range that dominates the rest of the data, which is a genuinely different atomization regime, not just "more of the same but noisier." This is a sharper, more honest finding than the vague train-test gap I reported first - the model's actual weak point is one specific, physically-explainable hardware condition, not general unreliability.

## Direct check: does excluding the 0.894mm orifice actually help?

Rather than assume, I tested with vs. without it in training entirely:
- With 0.894mm (full 336 rows): grouped-CV mean R2=0.694, std=0.209
- Without it (302 rows): grouped-CV mean R2=0.755, std=0.137

Both the mean and the stability improve with it excluded - this isn't just one noisy group among several equally-noisy ones, it's a genuinely different regime that measurably drags down generalization on the rest of the (more homogeneous) dataset. For a production model I'd either train two separate models per regime or add a feature that distinguishes them, rather than pool everything and accept the worse aggregate number.

## What I'm still unsure about

-
