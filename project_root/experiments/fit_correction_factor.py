"""
fit_correction_factor.py
========================
Implements the sigmoidal fitting for the delta-based correction factor K_delta(p, delta)
as shown in the slide.

Model:
    K_eff(p, delta) = K_Loeb(p) * K_delta(p, delta)
    K_delta(p, delta) = K_min + (K_max - K_min) / (1 + exp(b * (delta - delta_c)))

Where K_min, K_max, b, delta_c depend linearly on porosity p.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import argparse

# --- Constants ---
ALPHA_LOEB = 1.37
K_MATRIX = 1.0

def loeb_model(p, k_m=K_MATRIX, alpha=ALPHA_LOEB):
    """Classical Loeb model: K = k_m * (1 - alpha * p)."""
    return np.maximum(0.0, k_m * (1.0 - alpha * p))

def sigmoidal_correction(delta, k_min, k_max, b, delta_c):
    """Sigmoidal correction factor K_delta(delta). Always returns positive values."""
    result = k_min + (k_max - k_min) / (1 + np.exp(b * (delta - delta_c)))
    return np.maximum(0.0, result)  # Enforce K_delta >= 0

def full_model(p_delta, k_min, k_max, b, delta_c):
    """Combined model: K_eff = K_Loeb(p) * K_delta(delta). Always positive.
    Note: p_delta is passed as a tuple/array or we assume p is fixed for a single fit.
    """
    p, delta = p_delta
    result = loeb_model(p) * sigmoidal_correction(delta, k_min, k_max, b, delta_c)
    return np.maximum(0.0, result)  # Enforce K_eff >= 0

def generate_synthetic_data():
    """Generates synthetic data matching the trends in the slide for verification."""
    ps = [0.1, 0.2, 0.3]
    deltas = np.linspace(0.05, 1.0, 20)
    
    rows = []
    for p in ps:
        # Parameters depending linearly on p (approximate from visual trends)
        # K_min increases slightly with p? Actually delta->0 means connected cracks.
        # k_max should be ~1.0 (approaching Loeb if pores are isolated)
        k_min = 0.1 - 0.2 * p  # lower for higher porosity
        k_max = 1.0 - 0.1 * p
        b = 15.0 + 10.0 * p    # steeper for higher p
        delta_c = 0.15 + 0.3 * p # transition shifts right for higher p
        
        k_loeb = loeb_model(p)
        for d in deltas:
            k_corr = sigmoidal_correction(d, k_min, k_max, b, delta_c)
            k_eff = k_loeb * k_corr
            # Add some noise
            k_eff += np.random.normal(0, 0.01)
            rows.append({"Target_P": p, "Delta": d, "K_eff": k_eff})
            
    return pd.DataFrame(rows)

def recover_from_results(results_dir):
    """Parses individual directory results into a single DataFrame."""
    print(f"Scanning {results_dir} for results...")
    rows = []
    results_dir = Path(results_dir)
    for case_dir in results_dir.glob("P_*_Delta_*"):
        if not case_dir.is_dir():
            continue
            
        res_file = case_dir / "thermalCoeff_amitex.txt"
        if not res_file.exists():
            continue
            
        # Parse P and Delta from folder name
        parts = case_dir.name.split("_")
        try:
            p_val = float(parts[1])
            d_val = float(parts[3])
            
            data = np.loadtxt(res_file)
            if data.shape == (3, 3):
                k_eff = np.trace(data) / 3.0
            else:
                k_eff = 0.0
                
            rows.append({"Target_P": p_val, "Delta": d_val, "K_eff": k_eff})
        except Exception as e:
            print(f"Error parsing {case_dir.name}: {e}")
            
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Target_P", "Delta"])
        summary_path = results_dir / "keff_vs_delta.csv"
        df.to_csv(summary_path, index=False)
        print(f"Recovered {len(df)} points -> {summary_path}")
    return df

def fit_all(df, output_dir):
    """Fits the sigmoidal model to each porosity group."""
    results = []
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0.1: "steelblue", 0.2: "darkorange", 0.3: "forestgreen"}
    
    delta_fine = np.linspace(df["Delta"].min(), df["Delta"].max(), 200)
    
    for p, group in df.groupby("Target_P"):
        x = group["Delta"].values
        y = group["K_eff"].values
        k_loeb = loeb_model(p)
        
        # We fit K_delta = K_eff / K_Loeb
        y_corr = y / k_loeb
        
        # Initial guess: k_min, k_max, b (negative!), delta_c
        p0 = [0.2, 1.0, -3.0, 1.0] # k_min < k_max, b < 0 for correct transition
        # Bounds: k_min >= 0, k_max <= 1.1, b < 0, delta_c in [0, 4]
        bounds = ([0.0, 0.8, -20.0, 0.0], [0.95, 1.1, -0.5, 4.0])
        try:
            popt, _ = curve_fit(sigmoidal_correction, x, y_corr, p0=p0, bounds=bounds, maxfev=10000)
            k_min, k_max, b, delta_c = popt
            
            # Plotting
            col = colors.get(p, "black")
            ax.scatter(x, y, color=col, alpha=0.6, label=f"Data p={p}")
            ax.plot(delta_fine, k_loeb * sigmoidal_correction(delta_fine, *popt), 
                    color=col, linestyle="--", label=f"Fit p={p}")
            
            results.append({
                "p": p,
                "k_min": k_min,
                "k_max": k_max,
                "b": b,
                "delta_c": delta_c
            })
        except Exception as e:
            print(f"Failed to fit p={p}: {e}")

    ax.set_xlabel(r"$\delta$ (-)")
    ax.set_ylabel(r"$K_{eff}$ (W/m·K)")
    ax.set_title(r"Sigmoidal Fit: $K_{eff}$ vs $\delta$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = output_dir / "Sigmoidal_Fits.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved fits plot to {plot_path}")
    
    return pd.DataFrame(results)

def plot_parameters(res_df, output_dir):
    """Plots the fitted parameters vs porosity to check linear dependence."""
    params = ["k_min", "k_max", "b", "delta_c"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, p_name in enumerate(params):
        ax = axes[i]
        x = res_df["p"].values
        y = res_df[p_name].values
        
        ax.scatter(x, y, color="red")
        
        # Linear fit: y = alpha * p + beta
        z = np.polyfit(x, y, 1)
        poly = np.poly1d(z)
        ax.plot(x, poly(x), "b--", alpha=0.7)
        
        ax.set_title(f"{p_name} vs porosity")
        ax.set_xlabel("p")
        ax.set_ylabel(p_name)
        
        # Label with equation
        ax.text(0.05, 0.9, f"y = {z[0]:.2f}p + {z[1]:.2f}", transform=ax.transAxes)

    fig.tight_layout()
    param_plot_path = output_dir / "Parameters_vs_Porosity.png"
    fig.savefig(param_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved parameter analysis to {param_plot_path}")

def plot_contour(res_df, output_dir):
    """Generates the contour plot of K(p, delta)."""
    # Use the linear fits of parameters to extrapolate a surface
    ps = np.linspace(0.05, 0.4, 50)
    deltas = np.linspace(0.0, 1.0, 50)
    P, D = np.meshgrid(ps, deltas)
    
    # Linear fits for each parameter
    fits = {}
    for p_name in ["k_min", "k_max", "b", "delta_c"]:
        fits[p_name] = np.polyfit(res_df["p"], res_df[p_name], 1)
        
    def get_val(p_name, p_val):
        z = fits[p_name]
        return z[0] * p_val + z[1]
    
    K = np.zeros_like(P)
    for i in range(len(ps)):
        for j in range(len(deltas)):
            p = P[j, i]
            d = D[j, i]
            k_min = get_val("k_min", p)
            k_max = get_val("k_max", p)
            b = get_val("b", p)
            delta_c = get_val("delta_c", p)
            K[j, i] = loeb_model(p) * sigmoidal_correction(d, k_min, k_max, b, delta_c)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(P, D, K, levels=20, cmap="viridis")
    fig.colorbar(cp, label=r"$K_{eff}(p, \delta)$")
    ax.set_xlabel("Porosity p")
    ax.set_ylabel(r"Delta $\delta$")
    ax.set_title(r"Contour plot of $K(p, \delta)$")
    
    contour_path = output_dir / "K_eff_Contour.png"
    fig.savefig(contour_path, dpi=300)
    plt.close(fig)
    print(f"Saved contour plot to {contour_path}")

class KeffPredictor:
    """Uses linear fits of sigmoidal parameters to predict K_eff(p, delta)."""
    def __init__(self, res_df):
        self.fits = {}
        for p_name in ["k_min", "k_max", "b", "delta_c"]:
            self.fits[p_name] = np.polyfit(res_df["p"], res_df[p_name], 1)
            
    def get_params(self, p):
        return {name: np.poly1d(fit)(p) for name, fit in self.fits.items()}
    
    def predict(self, p, delta):
        p_vals = self.get_params(p)
        k_loeb = loeb_model(p)
        k_corr = sigmoidal_correction(delta, p_vals["k_min"], p_vals["k_max"], 
                                      p_vals["b"], p_vals["delta_c"])
        return k_loeb * k_corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to keff_vs_delta.csv")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")
    parser.add_argument("--output-dir", type=str, default="Results_Sigmoidal_Fit")
    parser.add_argument("--predict", nargs=2, type=float, metavar=("P", "DELTA"),
                        help="Predict K_eff for given porosity and delta")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parameters if they exist for prediction
    param_path = out_dir / "fitted_parameters.csv"
    if args.predict:
        if not param_path.exists():
            print(f"Error: {param_path} not found. Run fitting first.")
            return
        res_df = pd.read_csv(param_path)
        predictor = KeffPredictor(res_df)
        p_val, d_val = args.predict
        k_pred = predictor.predict(p_val, d_val)
        print(f"\nPrediction for p={p_val}, delta={d_val}:")
        print(f"  K_eff = {k_pred:.4f} W/m·K")
        print(f"  (Comparison: K_Loeb = {loeb_model(p_val):.4f})")
        return

    if args.synthetic:
        print("Generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv(out_dir / "synthetic_data.csv", index=False)
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        # Fallback to recovery from standard location
        sim_dir = Path("Results_Keff_vs_Delta")
        if sim_dir.exists():
            df = recover_from_results(sim_dir)
        else:
            print("No data found. Use --synthetic, provide --csv, or run run_keff_vs_delta.py first.")
            return

    if df.empty:
        print("Error: Dataset is empty.")
        return

    res_df = fit_all(df, out_dir)
    if not res_df.empty:
        plot_parameters(res_df, out_dir)
        plot_contour(res_df, out_dir)
        res_df.to_csv(out_dir / "fitted_parameters.csv", index=False)
        print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
