# Stage 1 - DrivAerNet++: design-space EDA. 23 geometric design parameters +
# CFD-derived drag/lift coefficients for 4165 real car designs.
#
# Credit: DrivAerNet++ (Elrefaie et al., NeurIPS 2024) - see README.md for full
# citation. CC BY-NC 4.0 - non-commercial use only.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/DrivAerNet_ParametricData.csv")
print("Shape:", data.shape)
print(data.columns.tolist())

PARAMS = [c for c in data.columns if c not in
          ["Experiment", "Average Cd", "Std Cd", "Average Cl", "Std Cl", "Average Cl_f", "Std Cl_f", "Average Cl_r", "Std Cl_r"]]
print(f"\n{len(PARAMS)} design parameters")

# %%
print(data[["Average Cd", "Average Cl"]].describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(data["Average Cd"], bins=40, edgecolor="black")
axes[0].set_xlabel("Average Cd (drag coefficient)")
axes[0].set_title("Drag coefficient distribution (4165 designs)")

axes[1].hist(data["Average Cl"], bins=40, edgecolor="black", color="orange")
axes[1].set_xlabel("Average Cl (lift coefficient)")
axes[1].set_title("Lift coefficient distribution")
plt.tight_layout()
plt.savefig("../results/stage1_cd_cl_distributions.png", dpi=150)
plt.show()

# %%
# Which single geometry parameter correlates most strongly with Cd, on its own
# - not the full model yet, just what the raw design space looks like.
corrs = data[PARAMS].corrwith(data["Average Cd"]).sort_values(key=abs, ascending=False)
print("\nTop 8 parameters by |correlation| with Cd:")
print(corrs.head(8))

fig, ax = plt.subplots(figsize=(8, 6))
corrs.head(10).sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Correlation with Average Cd")
ax.set_title("Strongest single-parameter correlations with drag coefficient")
plt.tight_layout()
plt.savefig("../results/stage1_top_correlations.png", dpi=150)
plt.show()

# %%
# Scatter for the single strongest correlate - is the relationship actually
# linear, or does it just have the highest linear correlation coefficient?
top_param = corrs.index[0]
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(data[top_param], data["Average Cd"], alpha=0.3, s=10)
ax.set_xlabel(top_param)
ax.set_ylabel("Average Cd")
ax.set_title(f"Cd vs. {top_param} (r={corrs.iloc[0]:.3f})")
plt.tight_layout()
plt.savefig("../results/stage1_strongest_param_scatter.png", dpi=150)
plt.show()

# %%
# Next: fill in stage1_learning_notes.md - does the top-correlated parameter
# make physical sense as a drag driver, and how linear does the scatter
# actually look vs. the correlation coefficient alone suggesting?
