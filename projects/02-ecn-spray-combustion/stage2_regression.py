# Stage 2 - predict lift-off length: linear baseline -> Random Forest -> XGBoost
#
# Credit: Engine Combustion Network (ECN), Sandia National Laboratories - see
# README.md for full citation and required acknowledgment.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

clean = pd.read_csv("ecn_clean_liftoff.csv")
FEATURES = ["oxycon", "Ta", "dens", "injP", "orifDiam"]
TARGET = "liftoff"
print(f"Dataset: {clean.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    clean[FEATURES], clean[TARGET], test_size=0.25, random_state=0
)
print(f"Train: {len(X_train)}, test: {len(X_test)}")

# %%
models = {
    "Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=0),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results[name] = {"model": model, "pred": pred, "mae": mae, "rmse": rmse, "r2": r2}
    print(f"{name}: MAE={mae:.2f}mm, RMSE={rmse:.2f}mm, R2={r2:.3f}")

# %%
# Sanity check against overfitting: also report TRAIN R2 - if train R2 is much
# higher than test R2, the model is memorizing the clustered conditions from
# Stage 1's EDA, not learning a generalizable relationship.
print("\nTrain R2 (overfitting check):")
for name, r in results.items():
    train_r2 = r2_score(y_train, r["model"].predict(X_train))
    print(f"  {name}: train R2={train_r2:.3f}, test R2={r['r2']:.3f}, gap={train_r2-r['r2']:.3f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(y_test, r["pred"], alpha=0.6)
    lims = [min(y_test.min(), r["pred"].min()), max(y_test.max(), r["pred"].max())]
    ax.plot(lims, lims, "k--", alpha=0.5, label="perfect prediction")
    ax.set_xlabel("Actual lift-off (mm)")
    ax.set_ylabel("Predicted lift-off (mm)")
    ax.set_title(f"{name}\nMAE={r['mae']:.2f}mm, R2={r['r2']:.3f}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/stage2_model_comparison.png", dpi=150)
plt.show()

# %%
# The train/test gap above uses one random split. A random split doesn't
# respect the fact that this table pools several distinct rigs/nozzle sizes
# (Stage 1's clustering finding) - so I tested this properly: GroupKFold by
# orifice diameter, which forces each fold to test on a nozzle size the model
# never trained on. This is a genuine generalization test, not just a
# different random seed.
from sklearn.model_selection import GroupKFold, cross_val_score

print("\nGrouped CV by orifice diameter (tests generalization to an unseen nozzle size):")
groups = clean["orifDiam"]
gkf = GroupKFold(n_splits=4)
for name, model in [("Random Forest", models["Random Forest"]), ("XGBoost", models["XGBoost"])]:
    scores = cross_val_score(model, clean[FEATURES], clean[TARGET], cv=gkf, groups=groups, scoring="r2")
    print(f"  {name}: per-fold R2={np.round(scores, 2)}, mean={scores.mean():.3f}, std={scores.std():.3f}")

# %%
# Make the rig/hardware condition explicit rather than averaged away: leave-one-
# orifice-group-out. For each orifice diameter with enough samples, train on
# every OTHER orifice diameter and test only on that one - the honest way to
# see how well this generalizes to a genuinely new nozzle size.
rig_results = []
for orif_val, grp in clean.groupby("orifDiam"):
    if len(grp) < 8:
        continue  # too few samples to evaluate meaningfully
    train_data = clean[clean["orifDiam"] != orif_val]
    test_data = grp
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                          reg_lambda=2, subsample=0.8, colsample_bytree=0.8, random_state=0)
    model.fit(train_data[FEATURES], train_data[TARGET])
    pred = model.predict(test_data[FEATURES])
    r2 = r2_score(test_data[TARGET], pred) if len(test_data) > 1 else float("nan")
    mae = mean_absolute_error(test_data[TARGET], pred)
    rig_results.append({"orifDiam": orif_val, "n": len(test_data), "held_out_R2": r2, "held_out_MAE": mae})
    print(f"  orifDiam={orif_val}mm (n={len(test_data)}): held-out R2={r2:.3f}, MAE={mae:.2f}mm")

rig_df = pd.DataFrame(rig_results)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(rig_df["orifDiam"].astype(str), rig_df["held_out_R2"], color="tab:orange")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Held-out orifice diameter (mm)")
ax.set_ylabel("R2 on that orifice size (trained on all others)")
ax.set_title("XGBoost generalization to an unseen nozzle size, per rig condition")
plt.tight_layout()
plt.savefig("results/stage2_rig_generalization.png", dpi=150)
plt.show()

# %%
# Next: fill in stage2_learning_notes.md - which model actually generalizes
# best (test R2, not train R2), and does the train/test gap match the
# clustering concern flagged in Stage 1?
