"""Render the extra-target results as supplementary tables without manual transcription."""
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
v = root / "validation"
original = pd.read_csv(v / "target_results.csv").set_index("target")
sens = pd.read_csv(v / "activity_sensitivity.csv").set_index("target")
lines = []
def table(caption, header, rows):
    n = len(header)
    lines.extend([r"\begin{table}[htbp]\centering\small",r"\caption{"+caption+"}",
                  r"\begin{tabular}{@{}l"+"r"*(n-1)+r"@{}}\toprule",
                  " & ".join(header)+r"\\\midrule"])
    for row in rows:
        lines.append(" & ".join(row)+r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}", ""])

def f(x, n=2):
    return f"{x:.{n}f}"

table(r"Validation across 11 additional targets. Each target contributes 1,200 test predictions. The contrast is farther minus nearer RMSE among lead-distant compounds. Both groups are standardized to the same distribution of lead-distance bins within each target, fold, and draw. Intervals resample scaffolds. The last columns give the weighted distances to the leads and to the full training fold, nearer/farther.", 
      ["Target","Near RMSE","Far RMSE",r"$\Delta$","95\\% interval",r"$d_{\rm lead}$",r"$d_{\rm train}$"],
      [[r.label,f(r.near_rmse), f(r.far_rmse),f(r.delta_rmse),"["+f(r.ci_low)+", "+f(r.ci_high)+"]",f(r.near_novelty,3)+"/"+f(r.far_novelty,3),f(r.near_distance)+"/"+f(r.far_distance)] for r in original.itertuples()])
table(r"Error ranking and selective prediction on the additional targets. Rank correlations ($\rho$) are with absolute out-of-fold error. The adjusted disagreement column conditions on training distance and lead distance, and adjusted lead distance conditions on training distance. Entries that depend on the starting leads are medians across 100 draws. RMSE is reported for all predictions and for the fifth with the lowest disagreement.",
      ["Target",r"$\rho_{\sigma}$",r"Adj. $\rho_{\sigma}$",r"$\rho_{\rm train}$",r"$\rho_{\rm lead}$",r"Adj. $\rho_{\rm lead}$","All RMSE","20\\% RMSE"],
      [[r.label, f(r.rho_sigma_error),f(r.partial_sigma),f(r.rho_distance_error),f(r.rho_novelty_error),f(r.partial_novelty),f(r.rmse_all),f(r.rmse_low20)] for r in original.itertuples()])
table(r"Post hoc restriction to common measured activity within the original fold and lead-distance strata. The retained fraction refers to compound assignments across repeated lead draws. The difference and its 95\% scaffold-bootstrap interval are conditional on observations retained in bins represented on both sides. This procedure uses measured activities, which are unavailable for a prospective candidate.",
      ["Target","Retained (\\%)","Near RMSE","Far RMSE",r"$\Delta$","95\\% interval"],
      [[original.loc[i,"label"],f(100*r.retained_fraction,1),f(r.near_rmse),f(r.far_rmse),f(r.delta_rmse),"["+f(r.ci_low,3)+", "+f(r.ci_high,3)+"]"] for i,r in sens.iterrows()])
(root / "validation_tables.tex").write_text("\n".join(lines)+"\n")
