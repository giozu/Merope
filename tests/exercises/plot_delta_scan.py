import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------
# Sigmoid model
# -----------------------------------------------------------
def sigmoid(delta, k_min, k_max, a, delta_c):
    """
    k(delta) = k_min + (k_max - k_min) / (1 + exp[-a (delta - delta_c)])
    """
    return k_min + (k_max - k_min) / (1 + np.exp(-a * (delta - delta_c)))


# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------
df = pd.read_csv("DeltaScan_Results/delta_scan_results.txt", sep="\t")

p_unique = sorted(df["p_target"].unique())


# -----------------------------------------------------------
# 1) Plot raw curves + fitted sigmoids
# -----------------------------------------------------------

plt.figure(figsize=(10,6))
fit_params = {}
all_residuals = {}

print("\n========================")
print("   SIGMOID FITS")
print("========================")

for p in p_unique:
    sub = df[df["p_target"] == p].sort_values("delta_ratio")

    x = sub["delta_ratio"].values
    y = sub["Kmean"].values

    if len(x) < 4:
        print(f"Not enough points to fit p={p}, skipping.")
        continue

    # Stable initial guesses
    k_min0 = y.min()
    k_max0 = y.max()
    a0 = 10
    delta_c0 = 0.5 * (x.min() + x.max())   # robust initial guess

    p0 = [k_min0, k_max0, a0, delta_c0]

    try:
        popt, pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=20000)
    except Exception as e:
        print(f"Fit failed for p={p}: {e}")
        continue

    fit_params[p] = popt

    # Smooth curve
    xfit = np.linspace(x.min(), x.max(), 400)
    yfit = sigmoid(xfit, *popt)

    # Plot raw + fit
    plt.plot(x, y, "o", label=f"p={p} data")
    plt.plot(xfit, yfit, "-", label=f"p={p} fit")

    # residuals
    residuals = y - sigmoid(x, *popt)
    all_residuals[p] = (x, residuals)

    print(f"\n---- Fit p={p} ----")
    print(f"k_min   = {popt[0]:.6f}")
    print(f"k_max   = {popt[1]:.6f}")
    print(f"a       = {popt[2]:.6f}")
    print(f"delta_c = {popt[3]:.6f}")

    rmse = np.sqrt(np.mean(residuals**2))
    print(f"RMSE    = {rmse:.6f}")

plt.xlabel("delta_ratio")
plt.ylabel("Kmean")
plt.grid(True)
plt.legend()
plt.title("K(delta): raw data + sigmoid fits")
plt.show()

# -----------------------------------------------------------
# Plot residuals for each p separately
# -----------------------------------------------------------

for p, (x_res, res_vals) in all_residuals.items():
    plt.figure(figsize=(7,5))
    plt.plot(x_res, res_vals, "o-")
    plt.axhline(0, color="black", lw=1)
    plt.xlabel("delta_ratio")
    plt.ylabel("Residuals")
    plt.title(f"Residuals for p = {p}")
    plt.grid(True)
    plt.show()



# -----------------------------------------------------------
# 2) Porosity consistency check
# -----------------------------------------------------------

plt.figure(figsize=(7,5))
for p in p_unique:
    sub = df[df["p_target"] == p]
    plt.plot(sub["delta_ratio"], sub["p_por_meas"], "o-", label=f"p_target={p}")

plt.xlabel("delta_ratio")
plt.ylabel("Measured porosity p_por_meas")
plt.title("Measured porosity as function of δ/lagR")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------------------------------------
# 3) 2D contour map K(p_meas, δ)
# -----------------------------------------------------------

P = df["p_por_meas"].values    # <-- porosità reale!
D = df["delta_ratio"].values
K = df["Kmean"].values

grid_p = np.linspace(min(P), max(P), 100)
grid_d = np.linspace(min(D), max(D), 100)
PP, DD = np.meshgrid(grid_p, grid_d)

KK = griddata((P, D), K, (PP, DD), method="cubic")

plt.figure(figsize=(8,6))
plt.contourf(PP, DD, KK, levels=30, cmap="viridis")
plt.colorbar(label="Kmean")
plt.xlabel("Measured porosity p_por_meas")
plt.ylabel("delta_ratio")
plt.title("K(p_meas, δ) contour map")

# add sampling points
plt.scatter(P, D, c="white", s=10)

plt.show()


# -----------------------------------------------------------
# 4) 3D scatter plot
# -----------------------------------------------------------

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(P, D, K, c=K, cmap="viridis", s=30)

ax.set_xlabel("p_por_meas")
ax.set_ylabel("delta_ratio")
ax.set_zlabel("Kmean")
plt.title("3D scatter: K(p_meas, δ)")
plt.show()
