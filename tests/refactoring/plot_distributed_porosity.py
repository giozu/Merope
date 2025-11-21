import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Distributed_Test/distributed_results.txt", sep=r"\s+")

p = df["p_por_meas"].values
k_sim = df["Kmean"].values

# -----------------------------------------------------------
# Theoretical curves
# -----------------------------------------------------------

k_m = 1.0
k_g = 1e-3

# Maxwell–Eucken (dispersed pores)
def maxwell_eucken(p, k_m, k_g):
    return k_m * (
        ((k_g + 2*k_m) - 2*p*(k_m - k_g)) /
        ((k_g + 2*k_m) +   p*(k_m - k_g))
    )

k_ME = maxwell_eucken(p, k_m, k_g)

# Loeb empirical (tunable exponent)
def loeb(p, beta):
    return k_m * (1 - beta*p)

k_loeb_137 = loeb(p, 1.37)
k_loeb_25  = loeb(p, 2.5)


# --------------------------------------------
# plot 1: k_eff vs porosity (original)
# --------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(p, k_sim, "o", label="Simulation", color="black")
plt.plot(p, k_ME, "-", label="Maxwell-Eucken")
plt.plot(p, k_loeb_137, "-", label="Loeb (1.37)")
plt.plot(p, k_loeb_25, "-", label="Loeb (2.5)")
plt.xlabel("Porosity")
plt.ylabel("k_eff")
plt.grid(True)
plt.legend()
plt.title("Distributed porosity: k_eff vs porosity")
plt.tight_layout()
plt.show()
