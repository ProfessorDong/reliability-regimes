#!/usr/bin/env python3
"""Figure 2: the two chemical reference sets as a plane, and the balance the weighting achieves.

Panel a is pure geometry from the archived per-compound predictions: every held-out compound
of the 11-target validation panel placed by its distance to the sampled starting leads and its
distance to its training fold. Because the leads are drawn from the training set, no compound
can lie above the identity line. Panel b reports the achieved balance of the weighted
comparison, taken from the frozen panel summary rather than recomputed here.

Run from the directory holding validation/ (the repository root or the manuscript directory).
"""
import gzip
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#245d79"
ORANGE = "#b44e25"
GRAY = "#a2acb5"
INK = "#20282d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#91999e", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRED = ROOT / "validation" / "predictions"
SUMMARY = json.loads((ROOT / "validation" / "summary.json").read_text())


def load():
    frames = []
    for path in sorted(PRED.glob("*.csv.gz")):
        with gzip.open(path, "rt") as handle:
            frames.append(pd.read_csv(handle, usecols=["novelty_draw0", "distance"]))
    table = pd.concat(frames, ignore_index=True)
    assert len(table) == SUMMARY["n_target_parent_records"], "row count disagrees with the summary"
    assert (table.distance <= table.novelty_draw0 + 1e-9).all(), "a compound sits above the identity line"
    return table


def panel_a(ax, table):
    lead, train = table.novelty_draw0.to_numpy(), table.distance.to_numpy()
    cut = float(np.quantile(lead, 2 / 3))
    ax.fill_between([0, 1], [0, 1], [1.12, 1.12], color="#eef1f3", zorder=0, linewidth=0)
    ax.text(0.455, 0.80, "no compound can lie here\n($d_\\mathrm{train}\\leq d_\\mathrm{lead}$ always)",
            fontsize=7.3, color="#7b858c", ha="center", va="center", zorder=1)
    ax.axvspan(cut, 1.0, color=BLUE, alpha=0.07, zorder=1, linewidth=0)
    inside = lead >= cut
    ax.scatter(lead[~inside], train[~inside], s=1.4, color=GRAY, alpha=0.28,
               linewidths=0, zorder=2, rasterized=True)
    ax.scatter(lead[inside], train[inside], s=1.4, color=BLUE, alpha=0.40,
               linewidths=0, zorder=3, rasterized=True)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.9, ls="--", zorder=4)
    ax.text((cut + 1.0) / 2, 1.055, "most lead-distant third\n(the compounds compared)",
            fontsize=7.3, color=BLUE, ha="center", va="center", zorder=5)
    ax.set_xlim(0.30, 1.0)
    ax.set_ylim(0.0, 1.16)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Distance to the starting leads", fontsize=8.5)
    ax.set_ylabel("Distance to the training set", fontsize=8.5)
    ax.set_title("a   A compound has two distances", loc="left", fontweight="bold",
                 fontsize=10, pad=10)


def panel_b(ax):
    means = SUMMARY["means"]
    rows = [("Starting leads", means["near_novelty"], means["far_novelty"]),
            ("Training set", means["near_distance"], means["far_distance"])]
    for y, (label, near, far) in zip((1.0, 0.0), rows):
        ax.hlines(y, min(near, far), max(near, far), color=GRAY, lw=1.6, zorder=2)
        coincident = abs(near - far) < 0.02
        ax.scatter([far], [y], edgecolor=ORANGE, facecolor="white",
                   s=96 if coincident else 33, linewidth=1.3, zorder=3)
        ax.scatter([near], [y], color=BLUE, s=26 if coincident else 34, zorder=4)
        if coincident:
            ax.annotate(f"{near:.3f} and {far:.3f}\ndiffer by {abs(far - near):.3f}",
                        xy=(far, y), xytext=(far - 0.105, y + 0.50), fontsize=7.6,
                        color="#5b656c", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8,
                                        shrinkA=0, shrinkB=4))
        else:
            ax.text(near, y + 0.20, f"{near:.3f}", ha="center", fontsize=7.6, color=BLUE)
            ax.text(far, y + 0.20, f"{far:.3f}", ha="center", fontsize=7.6, color=ORANGE)
    ax.text(means["near_novelty"], 1.0 - 0.42, "matched", fontsize=7.8, color="#5b656c",
            ha="center", va="center", style="italic")
    ax.text((means["near_distance"] + means["far_distance"]) / 2, -0.42, "separated",
            fontsize=7.8, color="#5b656c", ha="center", va="center", style="italic")
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.72, 1.72)
    ax.set_xlim(0.18, 0.90)
    ax.set_xlabel("Mean distance, averaged over targets", fontsize=8.5)
    ax.grid(axis="x", color="#e6e9eb", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], color=BLUE, s=34, label="Nearer training group")
    ax.scatter([], [], edgecolor=ORANGE, facecolor="white", s=33, linewidth=1.3,
               label="Farther training group")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.30, -0.20), frameon=False,
              fontsize=8, borderaxespad=0, ncol=2, columnspacing=1.2)
    ax.set_title("b   Balance of the comparison", loc="left",
                 fontweight="bold", fontsize=10, pad=10)


def main():
    table = load()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.3, 3.5),
                                      gridspec_kw={"width_ratios": [1, 1.04]})
    panel_a(left, table)
    panel_b(right)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.235, wspace=0.38)
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)   # the repository tracks no rendered figures, so it has no figures/
    out = out_dir / "geometry.pdf"
    fig.savefig(out, dpi=400)
    print(f"wrote {out}")
    print(f"  {len(table):,} compounds; lead-distance tercile "
          f"{np.quantile(table.novelty_draw0, 2/3):.3f}")


if __name__ == "__main__":
    main()
