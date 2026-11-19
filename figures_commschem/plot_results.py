"""Build the statistical figures directly from the two panels' source data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE = "#245d79"
ORANGE = "#b44e25"
GRAY = "#a2acb5"
INK = "#20282d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#91999e",
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def axis_title(ax, panel, title):
    ax.set_title(f"{panel}   {title}", loc="left", fontweight="bold", fontsize=10, pad=12)
    ax.tick_params(length=3)
    ax.grid(axis="x", color="#e6e9eb", linewidth=0.6)
    ax.set_axisbelow(True)


def structural():
    d = pd.read_csv(ROOT / "validation" / "target_results.csv")
    d = d.set_index("target").loc[["htr2a", "adora2a", "cnr2", "oprk1", "oprm1", "egfr", "kdr", "ache", "faah", "ptgs2", "herg"]]
    summary = json.loads((ROOT / "validation" / "summary.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.3, 4.8), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.21, top=0.9, wspace=0.21)
    y = np.arange(len(d))[::-1]
    a.hlines(y, d.near_rmse, d.far_rmse, color=GRAY, lw=1.6)
    a.scatter(d.near_rmse, y, color=BLUE, s=29, label="Nearer training group", zorder=3)
    a.scatter(d.far_rmse, y, edgecolor=ORANGE, facecolor="white", s=28, linewidth=1.3, label="Farther training group", zorder=3)
    a.set_yticks(y, d.label)
    a.set_xlabel("RMSE (pActivity)")
    a.set_xlim(0.50, 1.32)
    a.set_xticks([.6,.8,1,1.2])
    a.legend(loc="upper left", bbox_to_anchor=(-.12,-.17), frameon=False, fontsize=8, borderaxespad=0)
    b.axvline(0, color=GRAY, lw=.9)
    b.errorbar(d.delta_rmse,y,xerr=np.vstack([d.delta_rmse-d.ci_low,d.ci_high-d.delta_rmse]),fmt="o",ms=4.3,color=BLUE,lw=1.1,capsize=2)
    m=summary["delta_rmse_mean"]
    lo,hi=summary["working_target_ci"]
    b.errorbar(m,-1.45,xerr=[[m-lo],[hi-m]],fmt="D",ms=5,color=ORANGE,lw=1.4,capsize=2.5)
    a.text(0.51,-1.45,"Equal-target mean",ha="left",va="center",weight="bold",fontsize=8)
    b.set_yticks([])
    b.set_xlabel("Farther minus nearer RMSE (pActivity)")
    b.set_xlim(-.025,.66)
    b.set_xticks([0,.2,.4,.6])
    for ax in (a,b):
        ax.set_ylim(-2,10.7)
        ax.axhline(-.7,color="#c1c7cb",lw=.7)
    for axis,letter,title in [(a,"a","Error in the two groups"),(b,"b","Difference within each target")]:
        axis_title(axis,letter,title)
    fig.savefig(OUT / "cross_family.pdf", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def decisions():
    tp = json.loads((ROOT/"source_data"/"temporal_analysis.json").read_text())
    pp = json.loads((ROOT/"source_data"/"poolopt_analysis.json").read_text())["summary"]
    fig, (a,b) = plt.subplots(1,2,figsize=(7.3,3.65),gridspec_kw={"width_ratios":[1,1.05]})
    fig.subplots_adjust(left=.08,right=.99,bottom=.25,top=.84,wspace=.34)
    targets=["scd1","nk1r","drd2","drd3"]
    y=np.arange(4)[::-1]
    rho=np.array([tp[k]["spearman_sigma_err"] for k in targets])
    bounds=np.array([tp[k]["scaffold_cluster_bootstrap"]["spearman_sigma_err_ci95"] for k in targets])
    ctrl=np.array([tp[k]["control_random_same_size"]["spearman_sigma_err"] for k in targets])
    a.hlines(y,ctrl,rho,color=GRAY,lw=1)
    a.errorbar(rho,y,xerr=np.stack([rho-bounds[:,0],bounds[:,1]-rho]),fmt="s",ms=4.4,color=ORANGE,lw=1.2,capsize=2.5,label="Later chemistry")
    a.scatter(ctrl,y,s=31,facecolor="white",edgecolor=BLUE,linewidth=1.4,label="Size-matched random",zorder=4)
    a.axvline(0,ls=":",color=GRAY,lw=1)
    a.set_xlim(-.16,.50);a.set_ylim(-.55,3.55)
    a.set_yticks(y,["SCD-1","NK1R","DRD2","DRD3"])
    a.set_xlabel("Spearman correlation with absolute error",fontsize=8)
    a.legend(loc="upper left",bbox_to_anchor=(-.12,-.24),fontsize=8,frameon=False)
    axis_title(a,"a","Ranking error by disagreement")
    methods=["ucb","lcb","conformal"]
    mlabels=["Mean + SD","Mean − SD","Lower score"]
    fulltargets=["scd1","fads","nk1r","drd2","drd3"]
    offsets=np.linspace(-.15,.15,len(fulltargets))
    for j,m in enumerate(methods):
        dif=np.array([pp[k][m]["hits"]-pp[k]["greedy"]["hits"] for k in fulltargets])
        b.scatter(dif,np.full(len(dif),2-j)+offsets,edgecolor=BLUE,facecolor="white",s=22,linewidth=1.0,label="Individual target" if j==0 else None)
        ci=t.ppf(.975,len(dif)-1)*dif.std(ddof=1)/np.sqrt(len(dif))
        b.errorbar(dif.mean(),2-j-.23,xerr=ci,fmt="D",ms=4.2,color=ORANGE,capsize=2.5,lw=1.2,label="Equal-target mean, 95% CI" if j==0 else None)
    b.axvline(0,color=GRAY,ls=":",lw=1)
    b.set_yticks([2,1,0],mlabels,fontsize=8)
    b.set_ylim(-.65,2.45)
    b.set_xlabel("Additional top-percentile hits vs. mean ranking",fontsize=8)
    b.legend(loc="upper left",bbox_to_anchor=(-.1,-.24),fontsize=8,frameon=False)
    axis_title(b,"b","Hits recovered in 300 queries")
    fig.savefig(OUT/"decisions.pdf",bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


if __name__ == "__main__":
    structural()
    decisions()
