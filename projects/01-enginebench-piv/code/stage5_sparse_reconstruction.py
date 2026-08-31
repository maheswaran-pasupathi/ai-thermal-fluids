# Stage 5 - EngineBench LSP Small: sparse/gappy PIV -> full-field reconstruction
#
# Technique: "gappy POD" (Everson & Sirovich, 1995). Build a POD basis from
# complete training snapshots, then for a NEW snapshot with missing points,
# solve a least-squares fit of POD coefficients using only the KNOWN points,
# and use those coefficients to reconstruct the FULL field (including the
# gaps). This is the sparse-to-full reconstruction task from the curriculum's
# Stage 5 stretch goal.
#
# Credit: Baker et al., EngineBench, arXiv:2406.03325, 2024 - see README.md
# Required acknowledgment: "The TCC engine work has been funded by General Motors
# through the General Motors University of Michigan Automotive Cooperative
# Research Laboratory, Engine Systems Division."

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

h5_paths = glob.glob("/kaggle/input/**/*.h5", recursive=True) or glob.glob("../data/*.h5")
data_path = h5_paths[0]
CAD = "cad090"
TEST_SNAP_IDX = 1000  # held out from POD training basis
GAP_FRACTION = 0.30   # fraction of points randomly masked out ("missing" PIV data)
K = 100               # POD modes used for the gappy fit
SEED = 0

rng = np.random.default_rng(SEED)

# %%
with h5py.File(data_path, "r") as f:
    test_point = list(f.keys())[0]
    dset = f[test_point][CAD]
    x0, y0 = dset[0, :, 0], dset[0, :, 1]
    ny, nx = len(np.unique(y0)), len(np.unique(x0))
    gridx, gridy = np.meshgrid(np.unique(x0), np.unique(y0))

    u_all = dset[:, :, 2]
    v_all = dset[:, :, 3]

X = np.concatenate([u_all, v_all], axis=1)
n_points = u_all.shape[1]

# %%
# Train POD basis on every snapshot EXCEPT the held-out test snapshot.
train_idx = [i for i in range(X.shape[0]) if i != TEST_SNAP_IDX]
X_train = X[train_idx]
X_mean = X_train.mean(axis=0)
Xc = X_train - X_mean

_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Phi = Vt[:K, :]  # (K, n_features) POD basis vectors

# %%
# Take the held-out snapshot, randomly mask GAP_FRACTION of the spatial points
# (masking both u and v at the same physical points, like a real PIV dropout).
truth = X[TEST_SNAP_IDX]
n_mask_points = int(GAP_FRACTION * n_points)
masked_point_idx = rng.choice(n_points, size=n_mask_points, replace=False)

known_mask = np.ones(2 * n_points, dtype=bool)
known_mask[masked_point_idx] = False              # mask u at these points
known_mask[masked_point_idx + n_points] = False    # mask v at these points

# Gappy POD: solve least squares for POD coefficients using only KNOWN entries.
y_known = (truth - X_mean)[known_mask]
Phi_known = Phi[:, known_mask]
coeffs, *_ = np.linalg.lstsq(Phi_known.T, y_known, rcond=None)

reconstructed = X_mean + coeffs @ Phi

# %%
rel_error_full = np.linalg.norm(reconstructed - truth) / np.linalg.norm(truth)
rel_error_gaps_only = (
    np.linalg.norm(reconstructed[~known_mask] - truth[~known_mask])
    / np.linalg.norm(truth[~known_mask])
)
print(f"Gap fraction: {GAP_FRACTION:.0%}, POD modes used: {K}")
print(f"Relative error, full field: {rel_error_full:.1%}")
print(f"Relative error, GAP POINTS ONLY (the actual reconstruction task): {rel_error_gaps_only:.1%}")

# %%
truth_u = truth[:n_points].reshape(ny, nx)
recon_u = reconstructed[:n_points].reshape(ny, nx)

gappy_u = truth[:n_points].copy()
gappy_u[masked_point_idx] = np.nan
gappy_u = gappy_u.reshape(ny, nx)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
im0 = axes[0].pcolormesh(gridx, gridy, truth_u, cmap="RdBu_r", shading="auto")
axes[0].set_title("Truth (complete)")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(gridx, gridy, gappy_u, cmap="RdBu_r", shading="auto")
axes[1].set_title(f"Gappy input ({GAP_FRACTION:.0%} missing)")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(gridx, gridy, recon_u, cmap="RdBu_r", shading="auto")
axes[2].set_title(f"Gappy-POD reconstruction\n(gap-only rel. error={rel_error_gaps_only:.1%})")
fig.colorbar(im2, ax=axes[2])

im3 = axes[3].pcolormesh(gridx, gridy, recon_u - truth_u, cmap="RdBu_r", shading="auto")
axes[3].set_title("Error (recon - truth)")
fig.colorbar(im3, ax=axes[3])

fig.suptitle(f"Stage 5: sparse-to-full PIV reconstruction (gappy POD), {CAD}, snapshot {TEST_SNAP_IDX}")
plt.tight_layout()
plt.savefig("../results/stage5_sparse_reconstruction.png", dpi=150)
plt.show()

# %%
# Next: fill in stage5_learning_notes.md. This closes out Project 01 - Portfolio
# Card 01 (see project README) can be written once these notes and Days 1-4's
# notes are filled in.
