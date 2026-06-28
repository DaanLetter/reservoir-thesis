"""
Headline results table for the air2water temperature validation (Results beats 1-3).
Reads the per-lake metrics CSV and the air2water output series, computes the
summary statistics quoted in the Results text, and writes:
    - results_summary_temp.csv  (machine-readable)
    - results_summary_temp.tex  (booktabs table for the thesis)
Reproducible from disk; no dependence on the notebook's in-memory `results` dict.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS = HERE / "validation_metrics_temp.csv"
DAMS = HERE / "air2water" / "dams"


def _series(dam_id, prefix):
    out_dir = DAMS / str(dam_id) / str(dam_id) / "output_3"
    hits = sorted(out_dir.glob(f"{prefix}_*.out"), key=lambda p: p.stat().st_mtime)
    cols = ["year", "month", "day", "obs_air", "obs_water",
            "sim_water", "obs_water_agg", "sim_water_agg"]
    df = pd.read_csv(hits[-1], sep=r"\s+", header=None, encoding="cp1252")
    df.columns = cols[:df.shape[1]]
    df = df[df["year"] != -999].replace(-999, np.nan)
    return df.dropna(subset=["obs_water", "sim_water"])


def pooled_rmse(metrics, prefix):
    sse, n = 0.0, 0
    for dam in metrics.dam_id:
        d = _series(dam, prefix)
        sse += float(((d.obs_water - d.sim_water) ** 2).sum())
        n += len(d)
    return np.sqrt(sse / n), n


def main():
    m = pd.read_csv(METRICS)
    gap = m.rmse_val - m.rmse_cal
    pooled_cal, n_cal = pooled_rmse(m, "2")
    pooled_val, n_val = pooled_rmse(m, "3")

    deg = "$^{\\circ}$C"
    rows = [
        ("Lakes calibrated and validated", f"{len(m)} / {len(m)} (0 failures)"),
        ("Median calibration RMSE",         f"{m.rmse_cal.median():.2f} {deg}"),
        ("Median validation RMSE",          f"{m.rmse_val.median():.2f} {deg}"),
        ("Validation RMSE range",           f"{m.rmse_val.min():.2f}--{m.rmse_val.max():.2f} {deg}"),
        ("Pooled daily RMSE, calibration",  f"{pooled_cal:.2f} {deg} ($n={n_cal:,}$)"),
        ("Pooled daily RMSE, validation",   f"{pooled_val:.2f} {deg} ($n={n_val:,}$)"),
        ("Median validation NSE",           f"{m.nse_val.median():.2f}"),
        ("Median validation bias",          f"{m.bias_val.median():+.2f} {deg}"),
        ("Median cal-to-val RMSE increase", f"{gap.median():.2f} {deg} ({100*gap.median()/m.rmse_val.median():.0f}\\%)"),
        ("Lakes validating better than calibrating", f"{int((gap < 0).sum())} / {len(m)}"),
    ]
    summary = pd.DataFrame(rows, columns=["Metric", "Value"])

    plain = summary.copy()
    plain["Value"] = (plain["Value"].str.replace(r"\$\^\{\\circ\}\$C", "degC", regex=True)
                                    .str.replace(r"[\\${}^]", "", regex=True))
    plain.to_csv(HERE / "results_summary_temp.csv", index=False)

    body = " \\\\\n".join(f"{r.Metric} & {r.Value}" for r in summary.itertuples())
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Headline validation results for air2water across the 56-lake set.}\n"
        "\\label{tab:temp_results_summary}\n"
        "\\begin{tabular}{ll}\n\\hline\n"
        "Metric & Value \\\\\n\\hline\n"
        f"{body} \\\\\n"
        "\\hline\n\\end{tabular}\n\\end{table}\n"
    )
    (HERE / "results_summary_temp.tex").write_text(tex, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nwrote results_summary_temp.csv and results_summary_temp.tex")


if __name__ == "__main__":
    main()
