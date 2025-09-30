"""
plot_results.py

Reload results from summary_results.npz and make plots:
- 2D heatmap of K_mean
- 3D surface plot of K_mean
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------------------------

results_folder = "map_L_RVE_resolution"
npz_path = os.path.join(results_folder, "summary_results.npz")

data = np.load(npz_path, allow_pickle=True)

results = data["results"]   # (Ncases, 7) array
L_vals  = data["L_vals"]    # unique L_RVE values
a_vals  = data["a_vals"]    # unique a values

print("Loaded results from:", npz_path)
print("Results shape:", results.shape)
print("Columns: L_RVE, a, N_voxel, L_voxel, K_mean, Error, Porosity_calc")

# ---------------------------------------------------------------------------
# ORGANIZE DATA FOR PLOTTING
# ---------------------------------------------------------------------------

# Extract columns
L_RVE      = results[:,0]
a_param    = results[:,1]
K_mean     = results[:,4]
Error      = results[:,5]

# Make grid
L_vals = np.unique(L_RVE)
a_vals = np.unique(a_param)
K_map  = np.full((len(L_vals), len(a_vals)), np.nan)

for r in results:
    i = np.where(L_vals == r[0])[0][0]
    j = np.where(a_vals == r[1])[0][0]
    K_map[i, j] = r[4]

# ---------------------------------------------------------------------------
# 2D HEATMAP
# ---------------------------------------------------------------------------

plt.figure(figsize=(8,6))
plt.imshow(np.ma.masked_invalid(K_map), origin="lower", aspect="auto",
           extent=[min(a_vals), max(a_vals), min(L_vals), max(L_vals)],
           cmap="viridis")
plt.colorbar(label="K_mean")
plt.xlabel("Resolution parameter a")
plt.ylabel("RVE size L")
plt.title("Effective conductivity map")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "K_map_replot.png"))
plt.show()

# ---------------------------------------------------------------------------
# 3D SURFACE PLOT
# ---------------------------------------------------------------------------

A_grid, L_grid = np.meshgrid(a_vals, L_vals)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(A_grid, L_grid, K_map, cmap="viridis",
                       edgecolor="k", alpha=0.8)

ax.set_xlabel("Resolution parameter a")
ax.set_ylabel("RVE size L")
ax.set_zlabel("Effective conductivity K_mean")
ax.set_title("3D map of K_mean vs L_RVE and a")

fig.colorbar(surf, shrink=0.5, aspect=10, label="K_mean")

# Optional: show data points as scatter
ax.scatter(a_param, L_RVE, K_mean, color="r", s=30, label="Data points")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "K_map_3D_replot.png"))
plt.show()
