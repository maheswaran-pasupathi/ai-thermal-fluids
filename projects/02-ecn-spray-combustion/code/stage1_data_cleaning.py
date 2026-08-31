# Stage 1 - ECN diesel spray/combustion data: clean table, preserve units/metadata
#
# Data source: Engine Combustion Network (Sandia National Laboratories), the
# open experimental-data table behind their Diesel Spray Combustion search tool.
# Freely downloadable, no account required.
#
# Credit: Engine Combustion Network (ECN), Sandia National Laboratories, and the
# contributing research institutions whose experiments populate this table.
# https://ecn.sandia.gov/diesel-spray-combustion/experimental-data-search/
# Please cite the ECN and the original experimental papers (see refs column /
# fileBaseName per row) for any use of this data beyond a learning exercise.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The CSV has 2 metadata/units rows before the real data starts.
raw = pd.read_csv("../data/ecn_dieseldata.csv", skiprows=[1, 2], na_values=["-"])
print("Raw shape:", raw.shape)

# %%
# Core physical inputs (ambient/injection condition) and candidate targets (KPIs).
# Column names/units come directly from the ECN table header - verified against
# the raw CSV before writing any of this, not assumed.
CORE_INPUTS = ["oxycon", "Ta", "dens", "injP", "orifDiam"]
TARGETS = ["liftoff", "igndly", "liqlen", "jetpen"]

print("\nNon-missing counts per candidate target:")
for t in TARGETS:
    print(f"  {t}: {raw[t].notna().sum()} / {len(raw)}")

print("\nRows with ALL core inputs + target present, per target:")
for t in TARGETS:
    complete = raw.dropna(subset=CORE_INPUTS + [t])
    print(f"  {t}: {len(complete)} rows")

# %%
# Lift-off length has the most complete coverage (~339 rows with every core
# input present) - that's the target for this project, not picked arbitrarily.
#
# Gotcha found while cleaning: many numeric cells embed an HTML annotation link
# after the value, e.g. "141.1;<a href=...>fuel pressure vs time</a>" - naive
# pd.to_numeric() silently turns these into NaN and throws away real data
# (dropped 338 -> 34 rows before this fix). Extract the leading numeric token
# instead of assuming the whole cell is a clean number.
TARGET = "liftoff"


def extract_leading_number(series):
    return pd.to_numeric(series.astype(str).str.extract(r"^\s*(-?\d+\.?\d*)")[0], errors="coerce")


clean = raw.dropna(subset=CORE_INPUTS + [TARGET])[CORE_INPUTS + [TARGET]].copy()
for col in CORE_INPUTS + [TARGET]:
    clean[col] = extract_leading_number(clean[col])
clean = clean.dropna()

print(f"\nFinal clean table for target='{TARGET}': {clean.shape}")
print(clean.describe())

clean.to_csv("../data/ecn_clean_liftoff.csv", index=False)

# %%
# EDA: distribution of each input and the target, plus target vs. each input.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cols = CORE_INPUTS + [TARGET]
units = {"oxycon": "%", "Ta": "K", "dens": "kg/m3", "injP": "MPa", "orifDiam": "mm", "liftoff": "mm"}
for ax, col in zip(axes.flat, cols):
    ax.hist(clean[col], bins=20, edgecolor="black")
    ax.set_title(f"{col} ({units[col]})")
plt.tight_layout()
plt.savefig("../results/stage1_distributions.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(1, len(CORE_INPUTS), figsize=(4 * len(CORE_INPUTS), 4))
for ax, col in zip(axes, CORE_INPUTS):
    ax.scatter(clean[col], clean[TARGET], alpha=0.5, s=15)
    ax.set_xlabel(f"{col} ({units[col]})")
    ax.set_ylabel(f"{TARGET} (mm)")
fig.suptitle(f"Lift-off length vs. each input condition ({len(clean)} clean rows)")
plt.tight_layout()
plt.savefig("../results/stage1_liftoff_vs_inputs.png", dpi=150)
plt.show()

# %%
# Next: fill in stage1_learning_notes.md - what's the strongest visible
# relationship here, and does it make physical sense (higher ambient O2 should
# shorten lift-off length, for example)?
