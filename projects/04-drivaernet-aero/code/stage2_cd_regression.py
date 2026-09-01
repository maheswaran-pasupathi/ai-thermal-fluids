# Stage 2 - predict Cd from the 23 geometry parameters: linear baseline ->
# Random Forest -> XGBoost.
#
# Credit: DrivAerNet++ (Elrefaie et al., NeurIPS 2024) - see README.md for full
# citation. CC BY-NC 4.0 - non-commercial use only.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

data = pd.read_csv("../data/DrivAerNet_ParametricData.csv")
PARAMS = [c for c in data.columns if c not in
          ["Experiment", "Average Cd", "Std Cd", "Average Cl", "Std Cl", "Average Cl_f", "Std Cl_f", "Average Cl_r", "Std Cl_r"]]
TARGET = "Average Cd"

X_train, X_test, y_train, y_test = train_test_split(data[PARAMS], data[TARGET], test_size=0.2, random_state=0)
print(f"Train: {len(X_train)}, test: {len(X_test)}")

# %%
models = {
    "Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0),
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = {"model": m, "pred": pred, "mae": mae, "r2": r2}
    print(f"{name}: MAE={mae:.4f}, R2={r2:.3f}")

# %%
# For context: Cd's std in this dataset is ~0.022, and typical CFD-to-CFD
# repeatability for drag on a mesh-refinement study is often ~0.002-0.005 -
# a useful sanity check for whether the model's MAE is "good" in absolute terms.
print(f"\nCd std in dataset: {data[TARGET].std():.4f}")
print(f"Best model MAE as fraction of Cd std: {min(r['mae'] for r in results.values()) / data[TARGET].std():.1%}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(y_test, r["pred"], alpha=0.4, s=15)
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Actual Cd")
    ax.set_ylabel("Predicted Cd")
    ax.set_title(f"{name}\nMAE={r['mae']:.4f}, R2={r['r2']:.3f}")
plt.tight_layout()
plt.savefig("../results/stage2_cd_model_comparison.png", dpi=150)
plt.show()

# %%
# Next: fill in stage2_learning_notes.md - is the best model's error small
# enough to be useful for early-stage design screening, and how does that
# compare to real CFD run-to-run variability?
