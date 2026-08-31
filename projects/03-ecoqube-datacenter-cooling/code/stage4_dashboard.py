# Stage 4 - static dashboard: given a cooling setting, predict the hotspot
# temperature and say whether it's safe; given a load/safety target, recommend
# a setting. Wraps Stages 2-3 into one callable interface rather than a
# Streamlit app - a portfolio-card deliverable, same standard as the earlier
# stages, not a production tool.
#
# Credit: ECO-Qube EU project (CORDIS 956059) and contributing partners - see
# README.md for full citation.

# %%
import pandas as pd
import numpy as np
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
FEATURE_RANGES = {f: (data[f].quantile(0.02), data[f].quantile(0.98)) for f in FEATURES}

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
model.fit(data[FEATURES], data[TARGET])


def predict_hotspot(settings: dict, safe_limit_c: float = 33.8) -> dict:
    """Given a cooling setting (dict of the 5 FEATURES), predict max rack temp
    and whether it's within the safe limit. Warns if the setting is outside
    the surrogate's trained range - extrapolation isn't supported."""
    row = pd.DataFrame([settings])[FEATURES]
    out_of_range = [f for f in FEATURES if not (FEATURE_RANGES[f][0] <= settings[f] <= FEATURE_RANGES[f][1])]
    pred = model.predict(row)[0]
    return {
        "predicted_max_temp_C": round(float(pred), 2),
        "within_safe_limit": bool(pred <= safe_limit_c),
        "out_of_trained_range": out_of_range,
    }


def recommend_setting(safe_limit_c: float = 33.8, n_levels: int = 8) -> dict:
    """Grid search (same approach as Stage 3) for the minimum cooling-power
    setting meeting the safety constraint, within the trained range."""
    grid = {f: np.linspace(*FEATURE_RANGES[f], n_levels) for f in FEATURES}
    combos = pd.MultiIndex.from_product(grid.values(), names=FEATURES).to_frame(index=False)
    combos["predicted_max_temp"] = model.predict(combos[FEATURES])
    combos["cooling_power_proxy"] = (
        combos["airIRG2GroupStatusCoolOutput[KWh]"] + combos["airIRG2RDT2StatusEvaporatorFanSpeed[%]"] / 100
    )
    feasible = combos[combos["predicted_max_temp"] <= safe_limit_c]
    if len(feasible) == 0:
        return {"feasible": False}
    best = feasible.loc[feasible["cooling_power_proxy"].idxmin()]
    return {"feasible": True, **best.to_dict()}


# %%
# Demo: two realistic requests through the dashboard interface.
print("=== Query 1: 'is this current setting safe?' ===")
example_current = {
    "airIRG2GroupStatusAirFlowMetric[L/s]": 8.10,
    "airIRG2GroupStatusCoolOutput[KWh]": 3.0,
    "airIRG2RDT2StatusEvaporatorFanSpeed[%]": 77.8,
    "airIRG2RDT2StatusReturnAirTempMetric[C]": 24.6,
    "airIRG2RDT2StatusSupplyAirTempMetric[C]": 21.1,
}
print(predict_hotspot(example_current))

print("\n=== Query 2: 'what's the cheapest safe setting?' ===")
print(recommend_setting())

print("\n=== Query 3: an out-of-range setting (should flag it, not silently extrapolate) ===")
example_extreme = dict(example_current)
example_extreme["airIRG2RDT2StatusEvaporatorFanSpeed[%]"] = 150.0  # unrealistic, outside observed data
print(predict_hotspot(example_extreme))
