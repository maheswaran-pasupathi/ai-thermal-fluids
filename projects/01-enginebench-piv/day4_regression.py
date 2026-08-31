# Day 4 - EngineBench LSP Small: crank angle -> POD coefficients -> reconstructed
# MEAN field, via leave-one-crank-angle-out regression.
#
# Honesty note: this small dataset only has 5 distinct crank angles (090-270),
# so this is a small-sample regression, not a robust model. It demonstrates the
# reduced-order-surrogate IDEA (condition -> POD coeffs -> field), while being
# explicit that a real model would need the full EngineBench dataset (many more
# crank angles / operating points) to generalize. See Phase 2 backlog.
#
# Credit: Baker et al., EngineBench, arXiv:2406.03325, 2024 - see README.md
# Required acknowledgment: "The TCC engine work has been funded by General Motors
# through the General Motors University of Michigan Automotive Cooperative
# Research Laboratory, Engine Systems Division."

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob

h5_paths = glob.glob("/kaggle/input/**/*.h5", recursive=True) or glob.glob("enginebench_data/*.h5")
data_path = h5_paths[0]
CAD_DEG = {"cad090": 90, "cad135": 135, "cad180": 180, "cad225": 225, "cad270": 270}
HOLD_OUT = "cad180"  # middle value -> interpolation test, not extrapolation

# %%
# Load every crank angle, build a GLOBAL POD basis pooling all cycles from all
# crank angles (so coefficients are comparable across crank angles).
all_u, all_v, cad_labels = [], [], []
with h5py.File(data_path, "r") as f:
    test_point = list(f.keys())[0]
    for cad in CAD_DEG:
        dset = f[test_point][cad]
        x0, y0 = dset[0, :, 0], dset[0, :, 1]
        ny, nx = len(np.unique(y0)), len(np.unique(x0))
        gridx, gridy = np.meshgrid(np.unique(x0), np.unique(y0))
        all_u.append(dset[:, :, 2])
        all_v.append(dset[:, :, 3])
        cad_labels += [cad] * dset.shape[0]

U = np.concatenate(all_u, axis=0)
V = np.concatenate(all_v, axis=0)
X = np.concatenate([U, V], axis=1)
cad_labels = np.array(cad_labels)
print("Pooled snapshot matrix:", X.shape)

X_mean = X.mean(axis=0)
Xc = X - X_mean
Ub, S, Vt = np.linalg.svd(Xc, full_matrices=False)
K = 50  # keep 50 POD modes - enough for a reasonable mean-field reconstruction
coeffs = Ub[:, :K] * S[:K]  # (n_snapshots, K) POD coefficients per snapshot

# %%
# Train: crank angle (degrees) -> mean POD coefficients, using the 4 non-held-out
# crank angles. Target = mean coefficient vector per crank angle (we're predicting
# the MEAN field per condition, not individual-cycle variability - Day 2 showed
# that variability is real and not explainable from crank angle alone).
train_mask = cad_labels != HOLD_OUT
train_cads = np.array([CAD_DEG[c] for c in cad_labels[train_mask]]).reshape(-1, 1)
train_coeffs = coeffs[train_mask]

reg = LinearRegression()
reg.fit(train_cads, train_coeffs)

pred_coeff = reg.predict([[CAD_DEG[HOLD_OUT]]])[0]
pred_field = X_mean + pred_coeff @ Vt[:K, :]

actual_mask = cad_labels == HOLD_OUT
actual_mean_field = X[actual_mask].mean(axis=0)

n_points = U.shape[1]
pred_u = pred_field[:n_points].reshape(ny, nx)
actual_u = actual_mean_field[:n_points].reshape(ny, nx)
error_field = pred_u - actual_u
rel_error = np.linalg.norm(pred_field - actual_mean_field) / np.linalg.norm(actual_mean_field)
print(f"Held out: {HOLD_OUT} ({CAD_DEG[HOLD_OUT]} deg). Relative error vs actual mean field: {rel_error:.1%}")

# %%
# Baseline 1: naive nearest known crank angle.
nearest_cad = "cad135"
nearest_field = X[cad_labels == nearest_cad].mean(axis=0)
naive_error = np.linalg.norm(nearest_field - actual_mean_field) / np.linalg.norm(actual_mean_field)

# Improvement: crank angle isn't globally linear, but it IS locally smooth. Instead
# of fitting one global line across all 4 widely-spaced training angles, linearly
# interpolate between just the TWO NEAREST known crank angles (cad135, cad225).
c_low = coeffs[cad_labels == "cad135"].mean(axis=0)
c_high = coeffs[cad_labels == "cad225"].mean(axis=0)
w = (CAD_DEG[HOLD_OUT] - 135) / (225 - 135)
interp_coeff = (1 - w) * c_low + w * c_high
interp_field = X_mean + interp_coeff @ Vt[:K, :]
interp_u = interp_field[:n_points].reshape(ny, nx)
interp_error = np.linalg.norm(interp_field - actual_mean_field) / np.linalg.norm(actual_mean_field)

print(f"Naive baseline (just use {nearest_cad}'s mean field): relative error = {naive_error:.1%}")
print(f"Global linear regression (all 4 training points): relative error = {rel_error:.1%}")
print(f"Local interpolation (nearest neighbors only): relative error = {interp_error:.1%}")

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
im0 = axes[0].pcolormesh(gridx, gridy, actual_u, cmap="RdBu_r", shading="auto")
axes[0].set_title(f"Actual mean U - {HOLD_OUT}")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(gridx, gridy, pred_u, cmap="RdBu_r", shading="auto")
axes[1].set_title(f"Global regression\nrel. error = {rel_error:.1%}")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(gridx, gridy, interp_u, cmap="RdBu_r", shading="auto")
axes[2].set_title(f"Local interpolation (nearest neighbors)\nrel. error = {interp_error:.1%}")
fig.colorbar(im2, ax=axes[2])

im3 = axes[3].pcolormesh(gridx, gridy, interp_u - actual_u, cmap="RdBu_r", shading="auto")
axes[3].set_title("Local interp error\n(pred - actual)")
fig.colorbar(im3, ax=axes[3])

fig.suptitle(f"Day 4: predicting {HOLD_OUT} - global regression vs local interpolation vs actual")
plt.tight_layout()
plt.savefig("day4_regression_result.png", dpi=150)
plt.show()

# %%
# Next: fill in day4_learning_notes.md - local interpolation beat both the global
# regression and the naive baseline. Why would nearest-neighbor interpolation beat
# a "smarter" global regression here? What would change with the FULL EngineBench
# dataset (more crank angles / operating points) instead of this 5-point subset?
