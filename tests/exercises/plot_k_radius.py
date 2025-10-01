# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Load results from CSV
data = pd.read_csv("thermal_conductivity_porosity/summary_keff_r.csv")

# Print headers
print(data.columns.tolist())

plt.figure(figsize=(7,5))
markers = {0.1: "o", 0.2: "s", 0.3: "D"}
colors = {0.1: "green", 0.2: "orange", 0.3: "blue"}

porosity_values = sorted(data["Porosity"].unique())

for por in porosity_values:
    subset = data[(data["Porosity"] == por) & (data["Rule"] == "Voigt")]
    plt.plot(
        subset["R_pore"], subset["K_mean"],
        marker=markers.get(por, "x"),
        linestyle="--",
        color=colors.get(por, "black"),
        label=f"Porosity {por}"
    )

plt.xlim(left=0)
plt.xlabel("Radius")
plt.ylabel(r"$k_{\mathrm{eff}}$")
# plt.title("Inclusions radius vs effective conductivity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("thermal_conductivity_porosity/k_eff_vs_radius.png", dpi=300)
plt.show()
