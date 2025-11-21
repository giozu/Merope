import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Geometry_Scan/geometry_scan_full.txt", sep=r"\s+")

crit1 = df["crit1"].values
crit2 = df["crit2"].values
k = df["Kmean"].values

# -----------------------------------------------------------
# 1) K vs crit1
# -----------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(crit1, k, c='k')
plt.xlabel("crit1 = L_RVE / R_pore")
plt.ylabel("k_eff")
plt.title("K vs crit1")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 2) K vs crit2
# -----------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(crit2, k, c='k')
plt.xlabel("crit2 = R_pore / L_voxel")
plt.ylabel("k_eff")
plt.title("K vs crit2")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 3) 3D map
# -----------------------------------------------------------

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(crit1, crit2, k, c=k, cmap="viridis")

ax.set_xlabel("crit1 = L_RVE / R_pore")
ax.set_ylabel("crit2 = R_pore / L_voxel")
ax.set_zlabel("k_eff")
ax.set_title("3D map: k_eff(crit1, crit2)")

fig.colorbar(p, ax=ax, shrink=0.6)
plt.tight_layout()
plt.show()
