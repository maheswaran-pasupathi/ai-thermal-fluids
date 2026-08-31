# Stage 3 - use the Stage 2 surrogate to search for the minimum-cooling-power
# setting that keeps predicted max rack temperature under a safe limit.
#
# Honesty constraint carried over from Stage 2: the surrogate was trained on a
# narrow observed operating window (33.1-34.5C), so the search space below is
# deliberately bounded to that same window - NOT extrapolated to settings the
# model never saw, which would be an unsupported claim.
#
# Credit: ECO-Qube EU project (CORDIS 956059) and contributing partners - see
# README.md for full citation.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("../data/merged_surrogate_dataset.csv")
FEATURES = [
    "airIRG2GroupStatusAirFlowMetric[L/s]",
    "airIRG2GroupStatusCoolOutput[KWh]",
    "airIRG2RDT2StatusEvaporatorFanSpeed[%]",
    "airIRG2RDT2StatusReturnAirTempMetric[C]",
    "airIRG2RDT2StatusSupplyAirTempMetric[C]",
]
TARGET = "max_rack_temp"

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
model.fit(data[FEATURES], data[TARGET])

# %%
# Objective: minimize cooling power (proxy = coolOutput + fan speed, both
# real energy-cost drivers) subject to predicted max rack temp <= SAFE_LIMIT.
# Search grid is bounded to the actual observed range of each feature - the
# surrogate has no evidence outside that range.
SAFE_LIMIT_C = 33.8  # below the historical max of 34.5C, a real safety margin

grid = {}
for f in FEATURES:
    lo, hi = data[f].quantile(0.02), data[f].quantile(0.98)  # trim extreme outlier readings
    grid[f] = np.linspace(lo, hi, 8)

combos = pd.MultiIndex.from_product(grid.values(), names=FEATURES).to_frame(index=False)
combos["predicted_max_temp"] = model.predict(combos[FEATURES])
combos["cooling_power_proxy"] = (
    combos["airIRG2GroupStatusCoolOutput[KWh]"] + combos["airIRG2RDT2StatusEvaporatorFanSpeed[%]"] / 100
)

feasible = combos[combos["predicted_max_temp"] <= SAFE_LIMIT_C]
print(f"Feasible settings (predicted Tmax <= {SAFE_LIMIT_C}C): {len(feasible)} / {len(combos)}")

if len(feasible) > 0:
    best = feasible.loc[feasible["cooling_power_proxy"].idxmin()]
    print("\nRecommended setting (minimum cooling power meeting the safety constraint):")
    print(best)
else:
    print("No feasible setting found within the observed operating range at this safety limit.")

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(combos["cooling_power_proxy"], combos["predicted_max_temp"],
                 c=combos["predicted_max_temp"] <= SAFE_LIMIT_C, cmap="RdYlGn_r", alpha=0.5)
ax.axhline(SAFE_LIMIT_C, color="black", linestyle="--", label=f"Safety limit ({SAFE_LIMIT_C}C)")
if len(feasible) > 0:
    ax.scatter([best["cooling_power_proxy"]], [best["predicted_max_temp"]],
               color="blue", s=150, marker="*", label="Recommended setting", zorder=5)
ax.set_xlabel("Cooling power proxy (coolOutput kWh + fan speed/100)")
ax.set_ylabel("Predicted max rack temp (C)")
ax.set_title("Cooling-power vs. thermal-safety tradeoff (surrogate-based search)")
ax.legend()
plt.tight_layout()
plt.savefig("../results/stage3_optimization_tradeoff.png", dpi=150)
plt.show()

# %%
# Next: fill in stage3_learning_notes.md - how much does the recommended
# setting actually save vs. the historical average operating point, and is
# that saving believable given the surrogate's real accuracy from Stage 2?
