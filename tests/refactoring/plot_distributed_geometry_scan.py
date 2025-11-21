import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# ===========================================================
# LOAD DATA
# ===========================================================

df = pd.read_csv("Geometry_Scan/geometry_scan.txt", sep=r"\s+")

crit1 = df["crit1"].values
crit2 = df["crit2"].values
k = df["Kmean"].values
p_meas = df["p_por_meas"].values

# ===========================================================
# THEORETICAL REFERENCE (LOEB)
# ===========================================================

P_FIXED = 0.10
K_MATRIX = 1.0
beta = 1.37
k_loeb = K_MATRIX * (1 - beta * P_FIXED)

# ===========================================================
# INTERPOLATION GRID FOR CONTOURS
# ===========================================================

crit1_lin = np.linspace(min(crit1), max(crit1), 200)
crit2_lin = np.linspace(min(crit2), max(crit2), 200)
C1, C2 = np.meshgrid(crit1_lin, crit2_lin)

# interpolations
K_grid = griddata((crit1, crit2), k, (C1, C2), method='cubic')
P_grid = griddata((crit1, crit2), p_meas, (C1, C2), method='cubic')
Delta_grid = griddata((crit1, crit2), k - k_loeb, (C1, C2), method='cubic')

# converged value estimate
k_inf = np.mean(df[df["crit2"] > 4]["Kmean"])
Err_grid = griddata((crit1, crit2), np.abs(k - k_inf), (C1, C2), method='cubic')

# ===========================================================
# 1) K vs crit1 (grouped by constant crit2)
# ===========================================================

plt.figure(figsize=(9,6))
for c2 in sorted(df["crit2"].unique()):
    sub = df[df["crit2"] == c2]
    plt.plot(sub["crit1"], sub["Kmean"], "-o", label=f"crit2={c2:.2f}")

plt.hlines(k_loeb, min(crit1), max(crit1), colors='r', linestyles='--', label=f"Loeb β={beta}")
plt.xlabel("crit1 = L_RVE / R_pore")
plt.ylabel("k_eff")
plt.title("K vs crit1 (grouped by crit2)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================
# 2) K vs crit2 (grouped by constant crit1)
# ===========================================================

plt.figure(figsize=(9,6))
for c1 in sorted(df["crit1"].unique()):
    sub = df[df["crit1"] == c1]
    plt.plot(sub["crit2"], sub["Kmean"], "-o", label=f"crit1={c1:.2f}")

plt.hlines(k_loeb, min(crit2), max(crit2), colors='r', linestyles='--', label="Loeb")
plt.xlabel("crit2 = R_pore / L_voxel")
plt.ylabel("k_eff")
plt.title("K vs crit2 (grouped by crit1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================
# 3) 3D MAP: k_eff
# ===========================================================

fig = plt.figure(figsize=(11,7))
ax = fig.add_subplot(111, projection='3d')
p3 = ax.scatter(crit1, crit2, k, c=k, cmap='viridis')
ax.set_xlabel("crit1")
ax.set_ylabel("crit2")
ax.set_zlabel("k_eff")
ax.set_title("3D map: k_eff")
fig.colorbar(p3, ax=ax)
plt.tight_layout()
plt.show()

# ===========================================================
# 4) K contour map (2D)
# ===========================================================

plt.figure(figsize=(10,8))
cp = plt.contourf(C1, C2, K_grid, levels=30, cmap="viridis")
cs = plt.contour(C1, C2, K_grid, colors='k', linewidths=0.5)
plt.scatter(crit1, crit2, c='white', edgecolors='k')
plt.xlabel("crit1")
plt.ylabel("crit2")
plt.title("Contour map: k_eff(crit1, crit2)")
plt.colorbar(cp).set_label("k_eff")
plt.tight_layout()
plt.show()

# ===========================================================
# 5) measured porosity vs crit1
# ===========================================================

plt.figure(figsize=(9,6))
plt.scatter(crit1, p_meas, c='k')
plt.hlines(P_FIXED, min(crit1), max(crit1), colors='r', linestyles='--', label="target porosity")
plt.xlabel("crit1")
plt.ylabel("measured porosity")
plt.title("p_meas vs crit1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================
# 6) measured porosity vs crit2
# ===========================================================

plt.figure(figsize=(9,6))
plt.scatter(crit2, p_meas, c='k')
plt.hlines(P_FIXED, min(crit2), max(crit2), colors='r', linestyles='--', label="target porosity")
plt.xlabel("crit2")
plt.ylabel("measured porosity")
plt.title("p_meas vs crit2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================
# 7) Porosity contour map
# ===========================================================

plt.figure(figsize=(10,8))
cp = plt.contourf(C1, C2, P_grid, levels=30, cmap="plasma")
cs = plt.contour(C1, C2, P_grid, colors='k', linewidths=0.5)
plt.scatter(crit1, crit2, c='white', edgecolors='k')
plt.xlabel("crit1")
plt.ylabel("crit2")
plt.title("Contour map: p_meas(crit1, crit2)")
plt.colorbar(cp).set_label("measured porosity")
plt.tight_layout()
plt.show()

# ===========================================================
# 8) Delta-k = k - Loeb (contour)
# ===========================================================

plt.figure(figsize=(10,8))
cp = plt.contourf(C1, C2, Delta_grid, levels=30, cmap="coolwarm")
cs = plt.contour(C1, C2, Delta_grid, colors='k', linewidths=0.5)
plt.scatter(crit1, crit2, c='white', edgecolors='k')
plt.xlabel("crit1")
plt.ylabel("crit2")
plt.title("Δk = k_eff − k_Loeb")
plt.colorbar(cp).set_label("Δk")
plt.tight_layout()
plt.show()

# ===========================================================
# 9) Convergence map: |k − k∞|
# ===========================================================

plt.figure(figsize=(10,8))
cp = plt.contourf(C1, C2, Err_grid, levels=30, cmap="inferno")
cs = plt.contour(C1, C2, Err_grid, colors='white', linewidths=0.5)
plt.scatter(crit1, crit2, c='cyan', edgecolors='k')
plt.xlabel("crit1")
plt.ylabel("crit2")
plt.title(f"Convergence: |k − k∞| (k∞ ≈ {k_inf:.5f})")
plt.colorbar(cp).set_label("absolute error")
plt.tight_layout()
plt.show()
