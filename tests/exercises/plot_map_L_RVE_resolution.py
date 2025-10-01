"""
map_L_RVE_resolution.py

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

target_porosity = 0.1
results_folder = "map_L_RVE_resolution_p0_1"
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
plt.savefig(os.path.join(results_folder, "K_map_3D_plot.png"))
plt.show()

# ---------------------------------------------------------------------------
# ERROR PLOT (relative anisotropy)
# ---------------------------------------------------------------------------

RelErr_map = np.full((len(L_vals), len(a_vals)), np.nan)
for r in results:
    i = np.where(L_vals == r[0])[0][0]
    j = np.where(a_vals == r[1])[0][0]
    if r[4] > 0:   # K_mean > 0
        RelErr_map[i, j] = r[5] / r[4]

plt.figure(figsize=(8,6))
plt.imshow(np.ma.masked_invalid(RelErr_map), origin="lower", aspect="auto",
           extent=[min(a_vals), max(a_vals), min(L_vals), max(L_vals)],
           cmap="magma")
plt.colorbar(label="Relative anisotropy error (offdiag / K_mean)")
plt.xlabel("Resolution parameter a")
plt.ylabel("RVE size L")
plt.title("Relative error map")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "RelError_map.png"))
plt.show()

# ---------------------------------------------------------------------------
# 2D LINE PLOT: Relative anisotropy error vs resolution parameter a
# ---------------------------------------------------------------------------

RelErr = Error / K_mean  # relative error (offdiag / K_mean)

plt.figure(figsize=(8,6))

for i, L in enumerate(L_vals):
    mask = (L_RVE == L)
    a_sub = a_param[mask]
    E_sub = RelErr[mask]

    # Sort by a for clean curves
    idx = np.argsort(a_sub)
    a_sorted = a_sub[idx]
    E_sorted = E_sub[idx]

    plt.plot(a_sorted, E_sorted, marker="o", label=f"L_RVE={L}")

plt.xlabel("Resolution parameter a")
plt.ylabel("Relative anisotropy error (offdiag / K_mean)")
plt.title("Relative error vs resolution parameter")
plt.yscale("log")
plt.legend(title="RVE size L")
plt.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "RelError_vs_a.png"))
plt.show()

# ---------------------------------------------------------------------------
# 2D LINE PLOTS: K_mean vs resolution parameter (a), one curve per L_RVE
# ---------------------------------------------------------------------------

plt.figure(figsize=(8,6))

for i, L in enumerate(L_vals):
    mask = (L_RVE == L)
    a_sub = a_param[mask]
    K_sub = K_mean[mask]

    # Sort by a to get clean lines
    idx = np.argsort(a_sub)
    a_sorted = a_sub[idx]
    K_sorted = K_sub[idx]

    plt.plot(a_sorted, K_sorted, marker="o", label=f"L_RVE={L}")

plt.xlabel("Resolution parameter a")
plt.ylabel("Effective conductivity K_mean")
plt.title("K_mean vs resolution parameter")
plt.legend(title="RVE size L")
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "K_vs_a_lines.png"))
plt.show()
7

# ---------------------------------------------------------------------------
# 2D LINE PLOTS: Porosity vs resolution parameter (a), one curve per L_RVE
# ---------------------------------------------------------------------------

plt.figure(figsize=(8,6))

for i, L in enumerate(L_vals):
    mask = (L_RVE == L)
    a_sub = a_param[mask]
    P_sub = results[mask, 6]   # porosity_calc is column 6

    # Sort by a
    idx = np.argsort(a_sub)
    a_sorted = a_sub[idx]
    P_sorted = P_sub[idx]

    plt.plot(a_sorted, P_sorted, marker="o", label=f"L_RVE={L}")

plt.axhline(target_porosity, color="k", linestyle="--", label="Target porosity")

plt.xlabel("Resolution parameter a")
plt.ylabel("Calculated porosity")
plt.title("Porosity vs resolution parameter")
plt.legend(title="RVE size L")
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "Porosity_vs_a.png"))
plt.show()

# ---------------------------------------------------------------------------
# 2D LINE PLOTS: Porosity vs L_RVE, one curve per resolution a
# ---------------------------------------------------------------------------

plt.figure(figsize=(8,6))

for j, a in enumerate(a_vals):
    mask = (a_param == a)
    L_sub = L_RVE[mask]
    P_sub = results[mask, 6]

    idx = np.argsort(L_sub)
    L_sorted = L_sub[idx]
    P_sorted = P_sub[idx]

    plt.plot(L_sorted, P_sorted, marker="s", label=f"a={a}")

plt.axhline(target_porosity, color="k", linestyle="--", label="Target porosity")

plt.xlabel("RVE size L")
plt.ylabel("Calculated porosity")
plt.title("Porosity vs RVE size")
plt.legend(title="Resolution a")
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, "Porosity_vs_L.png"))
plt.show()
