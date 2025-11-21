import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================
# LOAD DATA
# ===========================================================

df = pd.read_csv("Interconnected_Test/interconnected_results.txt", sep=r"\s+")

p_target = df["p_target"].values
p_meas   = df["p_meas"].values
delta    = df["delta_phys"].values

# ===========================================================
# PLOT
# ===========================================================

plt.figure(figsize=(8,6))

# scatter colorato in base allo spessore del film
sc = plt.scatter(p_target, p_meas, c=delta, cmap="viridis", edgecolors="k")

plt.plot([min(p_target), max(p_target)],
         [min(p_target), max(p_target)],
         "--", color="red", label="y = x (perfetto)")

plt.xlabel("target porosity p_target")
plt.ylabel("measured porosity p_meas")
plt.title("Interconnected porosity: target vs measured")
plt.grid(True)
plt.legend()

cbar = plt.colorbar(sc)
cbar.set_label("delta_phys")

plt.tight_layout()
plt.show()
