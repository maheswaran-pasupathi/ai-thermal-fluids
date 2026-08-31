# Day 2 - EngineBench LSP Small: crank-angle / cycle-to-cycle variability (EDA)
#
# Builds on day1_explore_piv.py. For each crank angle, computes the mean, RMS
# and std (cycle-to-cycle variability) of velocity magnitude across all 1041
# cycles at that crank angle - not just a single snapshot like Day 1.
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
print("Using:", data_path)

# %%
with h5py.File(data_path, "r") as f:
    test_point = list(f.keys())[0]
    crank_angles = list(f[test_point].keys())
    print(f"Test point: {test_point}, crank angles: {crank_angles}")

    # grid shape from the first snapshot of the first crank angle
    first = f[test_point][crank_angles[0]]
    x0, y0 = first[0, :, 0], first[0, :, 1]
    ny, nx = len(np.unique(y0)), len(np.unique(x0))
    gridx, gridy = np.meshgrid(np.unique(x0), np.unique(y0))

# %%
# For each crank angle: load ALL cycles, compute mean/RMS/std of velocity magnitude.
stats = {}
with h5py.File(data_path, "r") as f:
    for cad in crank_angles:
        dset = f[test_point][cad]
        n = dset.shape[0]
        u_all = dset[:, :, 2].reshape(n, ny, nx)
        v_all = dset[:, :, 3].reshape(n, ny, nx)
        mag_all = np.sqrt(u_all**2 + v_all**2)

        stats[cad] = {
            "mean": mag_all.mean(axis=0),
            "rms": np.sqrt((mag_all**2).mean(axis=0)),
            "std": mag_all.std(axis=0),  # cycle-to-cycle variability at each point
            "n_cycles": n,
        }
        print(f"{cad}: {n} cycles, mean |V| over field = {mag_all.mean():.2f}, "
              f"mean cycle-to-cycle std = {mag_all.std(axis=0).mean():.2f}")

# %%
# Plot mean and cycle-to-cycle std (variability) side by side per crank angle.
fig, axes = plt.subplots(2, len(crank_angles), figsize=(4 * len(crank_angles), 8))
for i, cad in enumerate(crank_angles):
    im0 = axes[0, i].pcolormesh(gridx, gridy, stats[cad]["mean"], cmap="viridis", shading="auto")
    axes[0, i].set_title(f"{cad} - mean |V|")
    fig.colorbar(im0, ax=axes[0, i])

    im1 = axes[1, i].pcolormesh(gridx, gridy, stats[cad]["std"], cmap="magma", shading="auto")
    axes[1, i].set_title(f"{cad} - cycle-to-cycle std")
    fig.colorbar(im1, ax=axes[1, i])

fig.suptitle(f"Cycle-to-cycle variability across crank angles - {test_point} ({stats[crank_angles[0]]['n_cycles']} cycles each)")
plt.tight_layout()
plt.savefig("day2_cycle_variability.png", dpi=150)
plt.show()

# %%
# Summary: field-averaged variability per crank angle - a single number to compare
# how "repeatable" the flow is at each point in the cycle.
print("\nField-averaged cycle-to-cycle std of |V| by crank angle:")
for cad in crank_angles:
    print(f"  {cad}: {stats[cad]['std'].mean():.3f}")

# %%
# Next: fill in day2_learning_notes.md with your own read on which crank angle(s)
# show the most cycle-to-cycle variability, and why that might matter physically
# (e.g. combustion-relevant crank angles vs. intake-dominated ones).
