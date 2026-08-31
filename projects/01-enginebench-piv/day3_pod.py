# Day 3 - EngineBench LSP Small: POD/PCA of the velocity field
#
# Uses cad090 (highest cycle-to-cycle variability from Day 2) across all 1041
# cycles. Builds a snapshot matrix of [u, v] fields, runs POD via SVD, plots
# explained energy vs mode count, and reconstructs one snapshot with a few
# different numbers of modes to see the reconstruction-fidelity tradeoff.
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

h5_paths = glob.glob("/kaggle/input/**/*.h5", recursive=True) or glob.glob("enginebench_data/*.h5")
data_path = h5_paths[0]
CAD = "cad090"

# %%
# Load all cycles for one crank angle, build the (n_snapshots, n_features) matrix.
# Features = u and v stacked, i.e. each snapshot is one long vector [u_flat, v_flat].
with h5py.File(data_path, "r") as f:
    test_point = list(f.keys())[0]
    dset = f[test_point][CAD]
    n_snaps = dset.shape[0]

    x0, y0 = dset[0, :, 0], dset[0, :, 1]
    ny, nx = len(np.unique(y0)), len(np.unique(x0))
    gridx, gridy = np.meshgrid(np.unique(x0), np.unique(y0))

    all_data = dset[:, :, :]  # (n_snaps, n_points, 4)
    u_all = all_data[:, :, 2]  # (n_snaps, n_points)
    v_all = all_data[:, :, 3]

X = np.concatenate([u_all, v_all], axis=1)  # (n_snaps, 2*n_points)
print("Snapshot matrix shape:", X.shape)

# %%
# POD via SVD on the mean-subtracted data.
X_mean = X.mean(axis=0)
X_centered = X - X_mean

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
energy = S**2
explained_ratio = energy / energy.sum()
cumulative = np.cumsum(explained_ratio)

n_modes_90 = np.searchsorted(cumulative, 0.90) + 1
n_modes_99 = np.searchsorted(cumulative, 0.99) + 1
print(f"Modes needed for 90% energy: {n_modes_90}")
print(f"Modes needed for 99% energy: {n_modes_99}")
print(f"Total modes available: {len(S)}")

# %%
# Plot the energy/cumulative-energy curve.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(explained_ratio[:50], "o-")
axes[0].set_xlabel("Mode number")
axes[0].set_ylabel("Explained energy fraction (log scale)")
axes[0].set_title("POD energy spectrum (first 50 modes)")

axes[1].plot(cumulative[:50], "o-")
axes[1].axhline(0.90, color="gray", linestyle="--", label="90%")
axes[1].axhline(0.99, color="black", linestyle="--", label="99%")
axes[1].set_xlabel("Mode number")
axes[1].set_ylabel("Cumulative explained energy")
axes[1].set_title("Cumulative POD energy")
axes[1].legend()

plt.tight_layout()
plt.savefig("day3_pod_energy.png", dpi=150)
plt.show()

# %%
# Reconstruct ONE snapshot with varying mode counts and compare to the original.
snap_idx = 0
mode_counts = [1, 5, n_modes_90, n_modes_99]

def reconstruct(k):
    return X_mean + (U[snap_idx, :k] * S[:k]) @ Vt[:k, :]

fig, axes = plt.subplots(1, len(mode_counts) + 1, figsize=(4 * (len(mode_counts) + 1), 4))

n_points = u_all.shape[1]
original_u = X[snap_idx, :n_points].reshape(ny, nx)
im = axes[0].pcolormesh(gridx, gridy, original_u, cmap="RdBu_r", shading="auto")
axes[0].set_title("Original U")
fig.colorbar(im, ax=axes[0])

for i, k in enumerate(mode_counts):
    recon = reconstruct(k)
    recon_u = recon[:n_points].reshape(ny, nx)
    err = np.linalg.norm(recon - X[snap_idx]) / np.linalg.norm(X[snap_idx])
    im = axes[i + 1].pcolormesh(gridx, gridy, recon_u, cmap="RdBu_r", shading="auto")
    axes[i + 1].set_title(f"k={k} modes\nrel. error={err:.1%}")
    fig.colorbar(im, ax=axes[i + 1])

fig.suptitle(f"POD reconstruction of U, {CAD}, snapshot {snap_idx} - mode count vs fidelity")
plt.tight_layout()
plt.savefig("day3_pod_reconstruction.png", dpi=150)
plt.show()

# %%
# Next: fill in day3_learning_notes.md - how many modes did it actually take to
# get a visually convincing reconstruction vs. the 90%/99% energy thresholds,
# and what does that gap tell you about this flow's structure?
