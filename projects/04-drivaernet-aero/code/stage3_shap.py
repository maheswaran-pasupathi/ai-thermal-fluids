# Stage 3 - SHAP feature importance for the Cd model, checked against known
# automotive aerodynamics rather than trusted blindly.
#
# Credit: DrivAerNet++ (Elrefaie et al., NeurIPS 2024) - see README.md for full
# citation. CC BY-NC 4.0 - non-commercial use only.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

data = pd.read_csv("../data/DrivAerNet_ParametricData.csv")
PARAMS = [c for c in data.columns if c not in
          ["Experiment", "Average Cd", "Std Cd", "Average Cl", "Std Cl", "Average Cl_f", "Std Cl_f", "Average Cl_r", "Std Cl_r"]]
TARGET = "Average Cd"

X_train, X_test, y_train, y_test = train_test_split(data[PARAMS], data[TARGET], test_size=0.2, random_state=0)

model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)
model.fit(X_train, y_train)

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=PARAMS).sort_values(ascending=False)
print("Top 10 features by mean |SHAP value|:")
print(importance.head(10))

fig, ax = plt.subplots(figsize=(8, 6))
importance.head(12).sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Mean |SHAP value| (Cd)")
ax.set_title("XGBoost feature importance - drag coefficient")
plt.tight_layout()
plt.savefig("../results/stage3_shap_importance.png", dpi=150)
plt.show()

# %%
fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, max_display=12, show=False)
plt.tight_layout()
plt.savefig("../results/stage3_shap_summary.png", dpi=150)
plt.show()

# %%
# Physical plausibility check: direction of effect for the top few features,
# checked against real automotive aero knowledge, not just "is it important."
# Expected: diffuser angle and fender/wheel-arch geometry affect underbody and
# wheel-well flow separation - real, established drag drivers, not arbitrary.
print("\nDirection check (correlation between feature value and its own SHAP contribution):")
for f in importance.head(6).index:
    idx = PARAMS.index(f)
    corr = np.corrcoef(X_test[f], shap_values.values[:, idx])[0, 1]
    print(f"  {f}: {corr:+.3f}")

# %%
# Next: fill in stage3_learning_notes.md - do the top SHAP features match
# Stage 1's raw correlation ranking, and do the directions make aerodynamic
# sense (or reveal an interaction effect a single correlation couldn't show)?
