"""Post hoc diagnostic of chemically identical fingerprints across random folds.

The primary weights remain fixed. Among the test compounds with positive nearest-
training distance, each arm is renormalized within target. This diagnostic does
not refit the forest or reconstruct the original lead-novelty strata.
"""

import numpy as np
import pandas as pd

from prepare import ROOT, TARGETS


if __name__ == "__main__":
    rows = []
    keys = []
    for target in TARGETS:
        data = pd.read_csv(ROOT / "predictions" / f"{target}.csv.gz", float_precision="round_trip")
        keys.extend(data.parent_key)
        keep = data.distance > 0.0
        subset = data.loc[keep]
        e2 = (subset.prediction - subset.pActivity) ** 2
        near = np.sqrt(np.average(e2, weights=subset.near_weight))
        far = np.sqrt(np.average(e2, weights=subset.far_weight))
        rows.append(
            dict(
                target=target,
                n=len(data),
                n_exact_fp=int((~keep).sum()),
                n_retained=int(keep.sum()),
                near_rmse_excluding_exact_fp=near,
                far_rmse_excluding_exact_fp=far,
                delta=far - near,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "fingerprint_overlap.csv", index=False)
    nkeys = pd.Series(keys).value_counts()
    print(
        f"{len(nkeys)} distinct keys; {(nkeys>1).sum()} keys shared across targets. Excluded {frame.n_exact_fp.sum()} test compounds with fingerprint identity. Conditional mean contrast {frame.delta.mean():.4f} over {len(frame)} targets, positive on {(frame.delta>0).sum()}."
    )
