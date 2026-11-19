"""Build the two dated replication tables from the four target-specific results."""
from pathlib import Path

import pandas as pd

from snapshot import HERE, TARGETS

RESULTS = HERE / "results"
DISPLAY = {"PTGS2": "COX-2", "KDR": "VEGFR2", "FAAH": "FAAH", "hERG": "hERG"}


def write_table(filename, caption, label, format_spec, header, body, note):
    text = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\small",
        r"\begin{tabular}{" + format_spec + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *[row + r" \\" for row in body],
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{minipage}{0.98\linewidth}\footnotesize " + note + r"\end{minipage}",
        r"\end{table}",
    ]
    (HERE / filename).write_text("\n".join(text) + "\n")


def main():
    main_rows = pd.concat([pd.read_csv(RESULTS / f"{t}_row.csv") for t in TARGETS], ignore_index=True)
    lead_rows = pd.concat([pd.read_csv(RESULTS / f"{t}_lead_row.csv") for t in TARGETS], ignore_index=True)
    main_rows.to_csv(HERE / "summary.csv", index=False)
    lead_rows.to_csv(HERE / "lead_summary.csv", index=False)
    lines = []
    for r in main_rows.itertuples():
        count = f"{r.n_train:,}/{r.n_cal:,}/{r.n_test:,}"
        lines.append(f"{DISPLAY[r.target]} & {count} & {r.control_rmse:.2f} / {r.temporal_rmse:.2f} & "
                     f"{r.control_rho:.2f} / {r.temporal_rho:.2f} & {r.control_coverage:.3f} / {r.temporal_coverage:.3f}")
    write_table(
        "table.tex", "Temporal validation on four additional targets.", "tab:extension-temporal", "lrrrr",
        r"Target & Fit/cal./test & RMSE & Error rank $\rho$ & Coverage",
        lines, r"Each pair gives the mean of 32 size-matched random splits followed by the temporal result. "
        r"Activity is reported in negative log molar units. Error rank is the Spearman correlation between "
        r"tree disagreement and absolute error. Coverage is the fraction of observations in the nominal "
        r"90\% interval. Earlier labels use eligible measurements published before 2015. Evaluation uses "
        r"parents whose first eligible target activity was reported from 2015 onward.",
    )
    lines = []
    for r in lead_rows.itertuples():
        if not pd.notna(getattr(r, "valid_draws", None)):
            lines.append(f"{DISPLAY[r.target]} & -- & -- & -- & -- & --")
            continue
        lines.append(f"{DISPLAY[r.target]} & {r.median_pairs:.0f} & {r.delta:+.2f} & "
                     f"[{r.pct025:+.2f}, {r.pct975:+.2f}] & {r.positive*100:.0f} & "
                     f"{r.median_abs_seed_distance_diff:.3f}")
    write_table(
        "lead_table.tex", "Training neighbors and measured errors on later compounds dissimilar to historical leads.",
        "tab:extension-temporal-leads", "lrrrrr",
        r"Target & Pairs & $\Delta$RMSE & Middle 95\% & Positive (\%) & Lead mismatch",
        lines, r"Each of 100 draws uses ten active leads from the earlier proper-training set. "
        r"Within the most lead-dissimilar third of the later compounds, nearer and farther training "
        r"neighbors are compared within five bins of lead distance. The table gives the median number of pairs "
        r"and median farther-minus-nearer RMSE, the central 95\% range across draws, the percentage of "
        r"positive differences, and the median mean absolute within-pair difference in lead distance. "
        r"Repeated draws use the same measured compounds and describe sensitivity to the chosen leads.",
    )
    lines = []
    for r in main_rows.itertuples():
        cells = []
        for k, precision in [("rmse", 2), ("rho", 2), ("coverage", 3)]:
            lower = getattr(r, "temporal_"+k+"_lo")
            upper = getattr(r, "temporal_"+k+"_hi")
            cells.append(f"[{lower:.{precision}f}, {upper:.{precision}f}]")
        lines.append(" & ".join([DISPLAY[r.target], str(r.n_scaffolds)] + cells))
    write_table(
        "interval_table.tex", "Scaffold bootstrap intervals for the four additional temporal targets.",
        "tab:extension-temporal-intervals", "lrrrr",
        r"Target & Scaffold groups & RMSE & Error rank $\rho$ & Coverage",
        lines, r"Intervals are the 2.5th and 97.5th percentiles of 2,000 test-set bootstrap samples, "
        r"resampling complete Bemis--Murcko scaffolds. Acyclic structures are separate groups. "
        r"The historical model and calibration quantile are fixed. These intervals describe sampling "
        r"of later scaffolds, conditional on the retained historical sample.",
    )


if __name__ == "__main__":
    main()
