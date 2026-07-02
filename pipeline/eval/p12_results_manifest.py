import pandas as pd
from pathlib import Path
import sys

def main():
    print("="*60)
    print(" Compiling Results Manifest for Phase 12 ")
    print("="*60)
    
    out_dir = Path("results/p11_baselines")
    det_path = out_dir / "metrics_deterministic.csv"
    lit_path = out_dir / "literature_metrics.csv"
    
    if not det_path.exists():
        print(f"Error: {det_path} not found. Please run p11_baselines.py first.")
        sys.exit(1)
        
    df_det = pd.read_csv(det_path, index_col=0)
    
    # We may optionally append literature baselines if they are done
    if lit_path.exists():
        df_lit = pd.read_csv(lit_path, index_col=0)
        df_combined = pd.concat([df_det, df_lit])
    else:
        print("Note: literature_metrics.csv not found (maybe still training). Proceeding without it.")
        df_combined = df_det
        
    # Standardize columns to exactly what we want in the paper
    cols = ["Sharpe Ratio", "CAGR", "Max Drawdown", "Calmar Ratio"]
    df_manifest = df_combined[cols].copy()
    
    # You could dynamically pull latency from P13 here if available.
    # For now we'll add placeholder latency columns which can be manually populated or merged later.
    df_manifest["Mean Latency (ms)"] = ""
    df_manifest["P99 Latency (ms)"] = ""
    
    # Fill known values from our profiling (D27 report)
    if "MIDAS-NSE (Unified baseline)" in df_manifest.index:
        df_manifest.loc["MIDAS-NSE (Unified baseline)", "Mean Latency (ms)"] = "2.44"
        df_manifest.loc["MIDAS-NSE (Unified baseline)", "P99 Latency (ms)"] = "2.78"
        
    # Sort for the paper
    # (Optional: can sort by Sharpe)
    
    manifest_path = Path("results/results_manifest.csv")
    df_manifest.to_csv(manifest_path)
    
    print(f"Manifest successfully written to: {manifest_path}")
    print(df_manifest.to_string())

if __name__ == "__main__":
    main()
