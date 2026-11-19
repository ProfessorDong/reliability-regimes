"""Write the four-target chronological summary from held-out predictions and lead draws."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root = Path(__file__).resolve().parent
results = root / "temporal" / "results"
lines = [
    r"\begin{table}[htbp]\centering\small",
    r"\caption{Bioactivity predictions for compounds whose first eligible activity measurement on the indicated target was published in 2015 or later. RMSE, error-ranking correlation, and coverage use all sampled later parents. The final two columns summarize the comparison at similar lead distances over 100 historical lead draws. The difference is farther minus nearer RMSE and is reported as the median of the individual draw differences.}",
    r"\label{tab:datedvalidation}",
    r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
    r"Target & Later & RMSE & Error $\rho$ & 90\% coverage & $\Delta$RMSE & Positive draws\\\midrule",
]
for target,label in [("faah","FAAH"),("ptgs2","COX-2"),("kdr","VEGFR2"),("herg","hERG")]:
    p = pd.read_csv(results / f"{target}_future_predictions.csv.gz")
    draws = pd.read_csv(results / f"{target}_lead_draws.csv")
    error = p.activity-p.prediction
    coverage = (error.abs() <= p.adaptive_90_halfwidth).mean()
    rho = spearmanr(error.abs(),p.disagreement).statistic
    cells = [label, f"{len(p):,}", f"{np.sqrt(np.mean(error**2)):.2f}", f"{rho:.2f}", f"{coverage:.3f}",
             f"{draws.difference.median():+.2f}", f"{sum(draws.difference>0)}/{len(draws)}"]
    lines.append(" & ".join(cells) + r"\\")
lines.append(r"\bottomrule\end{tabular}\end{table}")
(root / "dated_main_table.tex").write_text("\n".join(lines)+"\n")
