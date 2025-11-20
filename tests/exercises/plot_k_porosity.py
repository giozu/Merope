import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("k_eff_porosity/summary.csv")

# Constants
k_matrix = 1.0
k_gas = 1e-3

# porosity
phi = np.linspace(0, 0.3, 100)

# Maxwell model
k_maxwell = k_matrix * (
    ((k_gas + 2*k_matrix) - 2*phi*(k_matrix - k_gas)) /
    ((k_gas + 2*k_matrix) +   phi*(k_matrix - k_gas))
)

# Loeb model
alpha_loeb = 1.37
k_loeb_1 = k_matrix * (1 - alpha_loeb * phi)
k_loeb_2 = k_matrix * (1 - 2.5 * phi)

# --- PLOT ---
plt.figure(figsize=(7,5))

# Plot simulation results (Voigt)
subset = data[data["Rule"] == "Voigt"]
plt.plot(subset["Porosity"], subset["K_mean"], "o", color="black", label="Simulation")

# Plot models
plt.plot(phi, k_maxwell, label="Maxwell-Eucken")
plt.plot(phi, k_loeb_1, label="Loeb (1.37)")
# plt.plot(phi, k_loeb_2, label="Loeb (2.5)")

plt.xlabel("Porosity")
plt.ylabel(r"$k_{\mathrm{eff}}$")
# plt.title("Classical models comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("k_eff_porosity/k_eff_vs_porosity_models.png", dpi=300)
plt.show()
