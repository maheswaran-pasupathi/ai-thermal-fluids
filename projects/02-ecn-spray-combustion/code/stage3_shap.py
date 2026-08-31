# Stage 3 - SHAP feature importance for the lift-off length model, checked
# against known combustion physics rather than trusted blindly.
#
# Credit: Engine Combustion Network (ECN), Sandia National Laboratories - see
# README.md for full citation and required acknowledgment.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

clean = pd.read_csv("../data/ecn_clean_liftoff.csv")
FEATURES = ["oxycon", "Ta", "dens", "injP", "orifDiam"]
TARGET = "liftoff"

X_train, X_test, y_train, y_test = train_test_split(
    clean[FEATURES], clean[TARGET], test_size=0.25, random_state=0
)

# Same regularized XGBoost config from Stage 2, not a fresh untested model.
model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                      reg_lambda=2, subsample=0.8, colsample_bytree=0.8, random_state=0)
model.fit(X_train, y_train)

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# %%
# Global feature importance: mean absolute SHAP value per feature.
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=FEATURES).sort_values(ascending=False)
print("Feature importance (mean |SHAP value|, mm of lift-off length):")
print(importance)

fig, ax = plt.subplots(figsize=(7, 4))
importance.sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Mean |SHAP value| (mm)")
ax.set_title("XGBoost feature importance - lift-off length")
plt.tight_layout()
plt.savefig("../results/stage3_shap_importance.png", dpi=150)
plt.show()

# %%
# Direction of effect per feature (SHAP summary/beeswarm), not just magnitude -
# this is what lets me check physical plausibility, not just "what mattered."
fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../results/stage3_shap_summary.png", dpi=150)
plt.show()

# %%
# Physical plausibility check: does the SIGN of each feature's effect match
# known combustion physics, not just "is it important"?
# Expected directions (from combustion chemistry / Stage 1's own EDA):
#   Ta (ambient temp)   UP -> lift-off DOWN (faster ignition kinetics)
#   oxycon (ambient O2) UP -> lift-off DOWN (faster reaction, more oxidizer)
#   dens (ambient density) UP -> lift-off DOWN (denser air, more mixing)
#   injP (injection pressure) UP -> lift-off UP (higher jet velocity pushes flame downstream)
#   orifDiam (orifice diameter) -> mixed/complex, no simple textbook sign
print("\nDirection check (Spearman correlation between feature value and its own SHAP value):")
for f in FEATURES:
    idx = FEATURES.index(f)
    corr = np.corrcoef(X_test[f], shap_values.values[:, idx])[0, 1]
    print(f"  {f}: correlation with its SHAP contribution = {corr:+.3f}")

# %%
# Next: fill in stage3_learning_notes.md - which feature directions matched
# expectations and which didn't, and what a mismatch would actually mean
# (wrong physics understanding vs. a real, more subtle interaction effect).
