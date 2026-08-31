# Stage 1 - ECO-Qube data center cooling: inspect CFD vs experimental data,
# and the raw sensor logs behind the cooling-optimization surrogate.
#
# Credit: ECO-Qube EU project (CORDIS 956059) and contributing partners - see
# README.md for full citation.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# %%
# Part A: CFD numerical result vs. experimental measurement, for both the
# original and retrofitted data-center cooling designs - real CFD validation,
# the same thing you'd do with any CFD model before trusting it.
exp = pd.read_csv("../data/exhaust_temp_experimental.csv")
exp.columns = ["height_U", "z_m", "prev_exp_C", "retro_exp_C", "z_m2"]

def load_cfd_profile(path):
    df = pd.read_csv(path)
    df = df.iloc[:, :2]
    df.columns = ["temp_C", "arc_length_m"]
    return df

cfd_prev = load_cfd_profile("../data/previousDesign/cfd_numerical_exhaust_profile.csv")
cfd_retro = load_cfd_profile("../data/retrofittedDesign/cfd_numerical_exhaust_profile.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].plot(cfd_prev["temp_C"], cfd_prev["arc_length_m"], color="red", label="CFD (numerical)")
axes[0].scatter(exp["prev_exp_C"], exp["z_m"], color="black", marker="v", label="Experimental")
axes[0].set_title("Previous design")
axes[0].set_xlabel("Temperature (C)")
axes[0].set_ylabel("Height (m)")
axes[0].legend()

axes[1].plot(cfd_retro["temp_C"], cfd_retro["arc_length_m"], color="red", label="CFD (numerical)")
axes[1].scatter(exp["retro_exp_C"], exp["z_m"], color="black", marker="s", label="Experimental")
axes[1].set_title("Retrofitted design")
axes[1].set_xlabel("Temperature (C)")
axes[1].legend()

fig.suptitle("CFD vs. experimental exhaust air temperature profile, by rack height")
plt.tight_layout()
plt.savefig("../results/stage1_cfd_vs_experiment.png", dpi=150)
plt.show()

# %%
# Part B: parse the raw sensor logs (not valid JSON - single-quoted Python
# dict literals, one per line, "{timestamp: [list of {sensor: value}]}").
def parse_netbotz_log(path):
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue  # skip the header line ("<ip> SERVER VALUES")
            entry = ast.literal_eval(line)
            (timestamp, readings), = entry.items()
            row = {"timestamp": timestamp}
            for r in readings:
                row.update(r)
            records.append(row)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

prev_rack = parse_netbotz_log("../data/previousDesign/netbotz_rack_temps.txt")
retro_rack = parse_netbotz_log("../data/retrofittedDesign/netbotz_rack_temps.txt")
print("Previous design rack sensors:", prev_rack.shape)
print("Retrofitted design rack sensors:", retro_rack.shape)
print("Sensor columns:", [c for c in prev_rack.columns if c != "timestamp"])

# %%
# Rack-back temperature by height (U position), averaged over the whole
# logging window - the actual vertical thermal profile inside the cabinet.
def height_profile(df):
    cols = [c for c in df.columns if "Cabin" in c]
    heights = {}
    for c in cols:
        u = int("".join(ch for ch in c.split("-")[-1] if ch.isdigit()))
        heights[u] = df[c][df[c] > 0].mean()  # drop the 0.0 placeholder readings
    return pd.Series(heights).sort_index()

prev_profile = height_profile(prev_rack)
retro_profile = height_profile(retro_rack)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(prev_profile.values, prev_profile.index, "o-", label="Previous design")
ax.plot(retro_profile.values, retro_profile.index, "s-", label="Retrofitted design")
ax.set_xlabel("Mean rack-back temperature (C)")
ax.set_ylabel("Rack unit height (U)")
ax.set_title("In-cabinet vertical temperature profile (from live sensors)")
ax.legend()
plt.tight_layout()
plt.savefig("../results/stage1_rack_temp_profile.png", dpi=150)
plt.show()

# %%
# Next: fill in stage1_learning_notes.md - does the sensor-derived profile
# agree with the CFD/experimental exhaust profile in Part A, and what does
# "door opened" vs "door closed" (previous vs retrofitted) physically change?
