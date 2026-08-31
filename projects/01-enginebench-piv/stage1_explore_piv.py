# Stage 1 - EngineBench LSP Small: explore structure, plot U/V/magnitude/vorticity
#
# Run this in a Kaggle notebook with the "enginebench-lsp-small" dataset attached as input.
# Reference tutorial from the dataset author: https://www.kaggle.com/code/samueljbaker/browsedata
# Reference data-loading code (verified structure used below): https://github.com/sambkr/EngineBench/blob/main/inpainting/utils/data_loading/michigan_piv.py
#
# Dataset and code credit: Samuel J. Baker, Michael A. Hobley, Isabel Scherl, Xiaohang Fang,
# Felix C. P. Leach, Martin H. Davy - Oxford TPSRG.
# Paper: Baker et al., "EngineBench: Flow Reconstruction in the Transparent Combustion Chamber III
# Optical Engine", arXiv:2406.03325, 2024.
# Dataset DOI: 10.34740/KAGGLE/DS/5000332
# Required acknowledgment per the source repo: "The TCC engine work has been funded by General
# Motors through the General Motors University of Michigan Automotive Cooperative Research
# Laboratory, Engine Systems Division."

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

# Kaggle mounts input datasets under /kaggle/input/<dataset-slug>/...
# find the .h5 file automatically instead of hardcoding a path
h5_paths = glob.glob("/kaggle/input/**/*.h5", recursive=True)
print("Found files:", h5_paths)
data_path = h5_paths[0]

# %%
# Step 1: look at the file structure before assuming anything about it.
# Top level = test points. Each test point group contains one dataset per crank angle.
with h5py.File(data_path, "r") as f:
    test_points = list(f.keys())
    print(f"{len(test_points)} test points, first few:", test_points[:5])

    first_tp = test_points[0]
    crank_angles = list(f[first_tp].keys())
    print(f"Crank angles in '{first_tp}':", crank_angles[:5], "...")

    first_cad = crank_angles[0]
    dset = f[first_tp][first_cad]
    print("Dataset shape (n_snapshots, n_points, 4):", dset.shape)
    print("Attributes:", dict(dset.attrs))

# %%
# Step 2: load one snapshot. Columns are [x, y, u, v] per michigan_piv.py.
with h5py.File(data_path, "r") as f:
    dset = f[first_tp][first_cad]
    snap = dset[0, ...]  # first snapshot at this test point / crank angle

    x, y, u, v = snap[:, 0], snap[:, 1], snap[:, 2], snap[:, 3]

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    ny, nx = len(unique_y), len(unique_x)

    gridu = u.reshape(ny, nx)
    gridv = v.reshape(ny, nx)
    gridx, gridy = np.meshgrid(unique_x, unique_y)

print("Grid shape:", gridu.shape)

# %%
# Step 3: velocity magnitude and a simple vorticity estimate (finite-difference curl).
magnitude = np.sqrt(gridu**2 + gridv**2)

dx = np.gradient(gridx, axis=1)
dy = np.gradient(gridy, axis=0)
dv_dx = np.gradient(gridv, axis=1) / dx
du_dy = np.gradient(gridu, axis=0) / dy
vorticity = dv_dx - du_dy

# %%
# Step 4: plot U, V, magnitude, vector field, vorticity.
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

im0 = axes[0, 0].pcolormesh(gridx, gridy, gridu, cmap="RdBu_r", shading="auto")
axes[0, 0].set_title("U (streamwise velocity)")
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].pcolormesh(gridx, gridy, gridv, cmap="RdBu_r", shading="auto")
axes[0, 1].set_title("V (cross-stream velocity)")
fig.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].pcolormesh(gridx, gridy, magnitude, cmap="viridis", shading="auto")
axes[0, 2].set_title("Velocity magnitude")
fig.colorbar(im2, ax=axes[0, 2])

step = 2  # subsample for a readable quiver plot
axes[1, 0].quiver(
    gridx[::step, ::step], gridy[::step, ::step],
    gridu[::step, ::step], gridv[::step, ::step],
)
axes[1, 0].set_title("Vector field (quiver)")
axes[1, 0].set_aspect("equal")

im3 = axes[1, 1].pcolormesh(gridx, gridy, vorticity, cmap="RdBu_r", shading="auto")
axes[1, 1].set_title("Vorticity (finite-difference estimate)")
fig.colorbar(im3, ax=axes[1, 1])

axes[1, 2].axis("off")

fig.suptitle(f"Test point: {first_tp} | Crank angle: {first_cad}")
plt.tight_layout()
plt.savefig("stage1_piv_snapshot.png", dpi=150)
plt.show()

# %%
# Step 5: compare a few crank angles from the SAME test point to see flow evolution.
# This feeds into Stage 2 (cycle/crank-angle variability) - just a first look here.
with h5py.File(data_path, "r") as f:
    sample_cads = crank_angles[:: max(1, len(crank_angles) // 4)][:4]
    fig, axes = plt.subplots(1, len(sample_cads), figsize=(4 * len(sample_cads), 4))
    for ax, cad in zip(axes, sample_cads):
        dset = f[first_tp][cad]
        snap = dset[0, ...]
        u, v = snap[:, 2], snap[:, 3]
        mag = np.sqrt(u**2 + v**2).reshape(ny, nx)
        im = ax.pcolormesh(gridx, gridy, mag, cmap="viridis", shading="auto")
        ax.set_title(cad)
    fig.suptitle(f"Velocity magnitude across crank angles - {first_tp}")
    plt.tight_layout()
    plt.savefig("stage1_crank_angle_comparison.png", dpi=150)
    plt.show()

# %%
# Next: fill in projects/01-enginebench-piv/stage1_learning_notes.md with your own
# physical observations and ML/data-concept notes before calling Stage 1 done.
