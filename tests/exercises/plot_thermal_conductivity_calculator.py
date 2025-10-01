import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------------
data = pd.read_csv("thermal_conductivity_calculator0_2/summary_0_2.csv")

# Each homogenization rule has its own style
markers = {
    "Voigt": "o",
    "Reuss": "s",
    "Smallest": "^",
    "Largest": "D",
}
colors = {
    "Voigt": "red",
    "Reuss": "orange",
    "Smallest": "blue",
    "Largest": "purple",
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(8,6))

for rule in data["Rule"].unique():
    subset = data[data["Rule"] == rule]
    plt.scatter(
        subset["N_voxel"], subset["K_mean"],
        label=rule,
        marker=markers.get(rule, "o"),
        color=colors.get(rule, "black"),
        s=60
    )

plt.xlabel("Number of voxels")
plt.ylabel(r"$k_{\mathrm{eff}}$")
plt.title("Homogenization rule")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("thermal_conductivity_calculator0_2/k_eff_vs_voxels.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(8,6))

for rule in data["Rule"].unique():
    subset = data[data["Rule"] == rule]
    plt.plot(
        subset["a"], subset["K_mean"],
        label=rule,
        marker=markers.get(rule, "o"),
        color=colors.get(rule, "black"),
        markersize=7,
        linestyle="--"   # linea tratteggiata con i marker
    )

plt.xlabel("Resolution parameter a")
plt.ylabel(r"$k_{\mathrm{eff}}$")
plt.title("Homogenization rule")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("thermal_conductivity_calculator0_2/k_eff_vs_a.png", dpi=300)
plt.show()