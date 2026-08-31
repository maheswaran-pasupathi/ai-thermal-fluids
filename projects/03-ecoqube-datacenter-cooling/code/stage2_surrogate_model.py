# Stage 2 - lightweight thermal KPI surrogate: predict max rack-back
# temperature from cooling-unit operational settings (fan speed, airflow,
# cool output, supply/return air temp), so an optimizer (Stage 3) doesn't
# need a full CFD run for every candidate setting.
#
# Credit: ECO-Qube EU project (CORDIS 956059) and contributing partners - see
# README.md for full citation.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def parse_log(path):
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            entry = ast.literal_eval(line)
            (timestamp, readings), = entry.items()
            row = {"timestamp": timestamp}
            for r in readings:
                row.update(r)
            records.append(row)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def max_rack_temp(netbotz_df):
    cols = [c for c in netbotz_df.columns if "Cabin" in c]
    # 0.0 readings are placeholder/missing, not real zero-temperature sensors
    valid = netbotz_df[cols].where(netbotz_df[cols] > 0)
    return valid.max(axis=1)

# %%
# Build one combined dataset from both designs - more data, and "design" is
# just another condition, not a reason to keep them artificially separate.
rows = []
for design, door in [("previous", "previousDesign"), ("retrofitted", "retrofittedDesign")]:
    cooler = parse_log(f"../data/{door}/inrow_cooler_metrics.txt")
    rack = parse_log(f"../data/{door}/netbotz_rack_temps.txt")
    rack["max_rack_temp"] = max_rack_temp(rack)

    merged = pd.merge_asof(
        cooler.sort_values("timestamp"), rack[["timestamp", "max_rack_temp"]].sort_values("timestamp"),
        on="timestamp", direction="nearest", tolerance=pd.Timedelta("15s"),
    )
    merged["design"] = design
    rows.append(merged)

data = pd.concat(rows, ignore_index=True).dropna()
print("Combined dataset:", data.shape)

FEATURES = [
    "airIRG2GroupStatusAirFlowMetric[L/s]",
    "airIRG2GroupStatusCoolOutput[KWh]",
    "airIRG2RDT2StatusEvaporatorFanSpeed[%]",
    "airIRG2RDT2StatusReturnAirTempMetric[C]",
    "airIRG2RDT2StatusSupplyAirTempMetric[C]",
]
TARGET = "max_rack_temp"
print(data[FEATURES + [TARGET]].describe())

# %%
X_train, X_test, y_train, y_test = train_test_split(
    data[FEATURES], data[TARGET], test_size=0.25, random_state=0
)

models = {"Linear": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)}
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = {"model": m, "pred": pred, "mae": mae, "r2": r2}
    print(f"{name}: MAE={mae:.2f}C, R2={r2:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(y_test, r["pred"], alpha=0.5)
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Actual max rack temp (C)")
    ax.set_ylabel("Predicted (C)")
    ax.set_title(f"{name}\nMAE={r['mae']:.2f}C, R2={r['r2']:.3f}")
plt.tight_layout()
plt.savefig("../results/stage2_surrogate_comparison.png", dpi=150)
plt.show()

# %%
data.to_csv("../data/merged_surrogate_dataset.csv", index=False)

# %%
# Next: fill in stage2_learning_notes.md - is this surrogate actually useful
# for Stage 3's optimization, or is the error too large relative to the
# safe-temperature margins that matter operationally?
