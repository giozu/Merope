import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------------------------------------
# Load dataframe
# -----------------------------------------------------------

df = pd.read_csv(
    "Result_0/aggregated_results.txt",
    sep="\t",
    skiprows=8
)

print(df.head())

df["PorDelta"] = df["PorDelta"].astype(float)
df["PorIntra"] = df["PorIntra"].astype(float)

# -----------------------------------------------------------
# 1) Scatter grezza
# -----------------------------------------------------------

plt.figure(figsize=(6,4))
plt.scatter(df["PorDelta"], df["Kmean"], s=20, alpha=0.7)
plt.xlabel("PorDelta (intergranular porosity)")
plt.ylabel("Kmean")
plt.title("Scatter of all simulations")
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# 2) K(delta) media su PorIntra
# -----------------------------------------------------------

grouped = df.groupby("PorDelta")["Kmean"].mean()
x = grouped.index.values
y = grouped.values

plt.figure(figsize=(6,4))
plt.plot(x, y, marker="o")
plt.xlabel("PorDelta")
plt.ylabel("Kmean (mean over PorIntra)")
plt.title("K(delta) averaged over intra-granular porosity")
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# 3) Heatmap PorDelta × PorIntra
# -----------------------------------------------------------

pivot = df.pivot_table(
    index="PorDelta",
    columns="PorIntra",
    values="Kmean"
)

pivot = pivot.sort_index().sort_index(axis=1)

plt.figure(figsize=(8,6))
plt.imshow(
    pivot.values,
    origin='lower',
    cmap='viridis',
    extent=[
        pivot.columns.min(), pivot.columns.max(),
        pivot.index.min(), pivot.index.max()
    ],
    aspect='auto'
)
plt.colorbar(label='k_eff')
plt.xlabel("PorIntra")
plt.ylabel("PorDelta")
plt.title("k_eff map for mixed porosity")
plt.show()

# -----------------------------------------------------------
# 4) Sigmoid fit for K(δ)
# -----------------------------------------------------------

def sigmoid(delta, k_min, k_max, a, delta_c):
    return k_min + (k_max - k_min) / (1 + np.exp(-a * (delta - delta_c)))

# initial guess
k_min0 = np.min(y)
k_max0 = np.max(y)
a0 = 10     # steepness guess
delta_c0 = x[np.argmin(np.gradient(y))]  # rough guess for transition

p0 = [k_min0, k_max0, a0, delta_c0]

# fit curve
popt, pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=10000)

k_min, k_max, a, delta_c = popt

print("------ Sigmoid fit parameters ------")
print(f"k_min   = {k_min:.6f}")
print(f"k_max   = {k_max:.6f}")
print(f"a       = {a:.6f}")
print(f"delta_c = {delta_c:.6f}")

# compute fit curve
x_fit = np.linspace(min(x), max(x), 200)
y_fit = sigmoid(x_fit, *popt)

# plot
plt.figure(figsize=(7,5))
plt.plot(x, y, "o", label="Data (mean)", markersize=8)
plt.plot(x_fit, y_fit, "-", label="Sigmoid fit", linewidth=2)

plt.xlabel("PorDelta")
plt.ylabel("Kmean")
plt.title("Sigmoid fit of K(delta)")
plt.grid(True)
plt.legend()
plt.show()
