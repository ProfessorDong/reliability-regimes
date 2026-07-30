"""Pseudo-prospective temporal validation (Tier-3).

Random cross-validation asks how a model behaves on compounds drawn from the same pool
it was fitted on. A prospective user is in a different position: the model is fitted on
what is known today and applied to chemistry that has not been published yet. The
re-curated ChEMBL files carry a publication year for every compound, so we can emulate
that directly.

Protocol, per target:
  * train on all compounds whose median publication year is < CUT
  * evaluate on compounds published in or after CUT, which the model has never seen and
    which were not available when it was fitted
  * grouping is on the standardized parent InChIKey, so no structure crosses the split
  * the reliability score, nearest-training distance and conformal calibration are all
    computed using ONLY pre-CUT data

We then ask the question that matters for deployment: on genuinely future compounds,
does ensemble disagreement still rank error, and do conformal intervals calibrated on
the past still cover the future? Coverage loss under temporal shift is the honest
measure of how far an exchangeability assumption can be pushed.

Writes temporal_analysis.json.

    python -m world_model.run_temporal_v1 --cut 2015
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.run_applicability_v1 import max_tanimoto_to_set, partial_spearman  # noqa: E402
from world_model.oracle import featurize  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'chembl_v2')
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'temporal_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
EPS = 1e-3


def load(target):
    rows = list(csv.DictReader(open(os.path.join(DATA, f'{target}_chembl_v2.csv'))))
    rows = [r for r in rows if r['year_median']]
    smi = [r['SMILES'] for r in rows]
    y = np.array([float(r['pAct']) for r in rows])
    yr = np.array([int(r['year_median']) for r in rows])
    return smi, y, yr


def conformal_q(scores, alpha=0.1):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else float(np.max(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--cut', type=int, default=2015)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    out = {'cut_year': args.cut,
           'protocol': 'train on compounds published before CUT, evaluate on CUT and later; '
                       'grouping on standardized parent InChIKey; calibration uses pre-CUT only'}
    P = {k: [] for k in ('err', 'sig', 'dtr', 'cov')}
    for tgt in args.targets:
        smi, y, yr = load(tgt)
        X = featurize(smi)
        tr = np.where(yr < args.cut)[0]
        te = np.where(yr >= args.cut)[0]
        if len(tr) < 100 or len(te) < 50:
            print(f"  {tgt}: insufficient split ({len(tr)}/{len(te)}), skipped"); continue
        # calibration split drawn from the PAST only
        rng = np.random.default_rng(0); perm = rng.permutation(len(tr))
        ncal = max(50, len(tr) // 4)
        cal, ptr = tr[perm[:ncal]], tr[perm[ncal:]]
        rf = RandomForestRegressor(**RF_KW).fit(X[ptr], y[ptr])
        def pred(idx):
            Q = np.stack([t.predict(X[idx]) for t in rf.estimators_], 0)
            return Q.mean(0), Q.std(0)
        yc, sc = pred(cal); yt, st_ = pred(te)
        err_c = np.abs(y[cal] - yc); err_t = np.abs(y[te] - yt)
        q_std = conformal_q(err_c, 0.1)
        q_ada = conformal_q(err_c / (sc + EPS), 0.1)
        cov_std = err_t <= q_std
        cov_ada = err_t <= q_ada * (st_ + EPS)
        dtr = 1.0 - max_tanimoto_to_set(X[te], X[ptr])
        rho_sig = stats.spearmanr(st_, err_t)
        rho_dtr = stats.spearmanr(dtr, err_t)
        part = partial_spearman(err_t, st_, dtr)
        # risk-coverage on the future compounds
        order = np.argsort(st_); e = err_t[order]
        rc = {f'{c:.1f}': float(np.sqrt(np.mean(e[:max(1, int(c * len(e)))] ** 2)))
              for c in (0.2, 0.4, 0.6, 0.8, 1.0)}
        out[tgt] = dict(
            n_train=int(len(ptr)), n_cal=int(len(cal)), n_test=int(len(te)),
            train_years=[int(yr[tr].min()), int(yr[tr].max())],
            test_years=[int(yr[te].min()), int(yr[te].max())],
            rmse_test=float(np.sqrt(np.mean(err_t ** 2))),
            spearman_sigma_err=float(rho_sig[0]), p_sigma_err=float(rho_sig[1]),
            spearman_dtr_err=float(rho_dtr[0]),
            partial_err_sigma_given_dtr=part[0],
            conformal_coverage_standard=float(cov_std.mean()),
            conformal_coverage_adaptive=float(cov_ada.mean()),
            conformal_width_standard=float(2 * q_std),
            conformal_width_adaptive_median=float(np.median(2 * q_ada * (st_ + EPS))),
            risk_coverage_rmse=rc,
        )
        # SIZE-MATCHED RANDOM CONTROL: same n_train / n_cal / n_test, drawn at random.
        # If the temporal degradation were only a sample-size effect, this control would
        # degrade equally. Repeated over 5 draws.
        ctl = {'rho': [], 'rmse': [], 'cov': []}
        allidx = np.arange(len(y))
        for rep in range(5):
            r2 = np.random.default_rng(100 + rep)
            pe = r2.permutation(allidx)
            c_te = pe[:len(te)]; c_cal = pe[len(te):len(te) + len(cal)]
            c_ptr = pe[len(te) + len(cal):len(te) + len(cal) + len(ptr)]
            rfc = RandomForestRegressor(**RF_KW).fit(X[c_ptr], y[c_ptr])
            def pc(idx):
                Q = np.stack([t.predict(X[idx]) for t in rfc.estimators_], 0)
                return Q.mean(0), Q.std(0)
            yc2, sc2 = pc(c_cal); yt2, st2 = pc(c_te)
            e_c = np.abs(y[c_cal] - yc2); e_t = np.abs(y[c_te] - yt2)
            qa2 = conformal_q(e_c / (sc2 + EPS), 0.1)
            ctl['rho'].append(float(stats.spearmanr(st2, e_t)[0]))
            ctl['rmse'].append(float(np.sqrt(np.mean(e_t ** 2))))
            ctl['cov'].append(float(np.mean(e_t <= qa2 * (st2 + EPS))))
        out[tgt]['control_random_same_size'] = dict(
            spearman_sigma_err=float(np.mean(ctl['rho'])),
            rmse=float(np.mean(ctl['rmse'])),
            conformal_coverage_adaptive=float(np.mean(ctl['cov'])), n_reps=5)
        for k, v in (('err', err_t), ('sig', st_), ('dtr', dtr), ('cov', cov_ada)):
            P[k].append(v)
        o = out[tgt]
        print(f"  {tgt}: train {o['n_train']} ({o['train_years'][0]}-{o['train_years'][1]}) "
              f"-> test {o['n_test']} ({o['test_years'][0]}-{o['test_years'][1]})  "
              f"RMSE {o['rmse_test']:.2f}  sig->err {o['spearman_sigma_err']:+.3f}  "
              f"partial|d {o['partial_err_sigma_given_dtr']:+.3f}  "
              f"90% coverage: std {o['conformal_coverage_standard']:.3f} "
              f"adaptive {o['conformal_coverage_adaptive']:.3f}  "
              f"RC {rc['0.2']:.2f}->{rc['1.0']:.2f}", flush=True)
        c = out[tgt]['control_random_same_size']
        print(f"        size-matched RANDOM control: RMSE {c['rmse']:.2f}  "
              f"sig->err {c['spearman_sigma_err']:+.3f}  90% coverage {c['conformal_coverage_adaptive']:.3f}",
              flush=True)
    if P['err']:
        err = np.concatenate(P['err']); sig = np.concatenate(P['sig'])
        dtr = np.concatenate(P['dtr']); cov = np.concatenate(P['cov'])
        order = np.argsort(sig); e = err[order]
        rc = {f'{c:.1f}': float(np.sqrt(np.mean(e[:max(1, int(c * len(e)))] ** 2)))
              for c in (0.2, 0.4, 0.6, 0.8, 1.0)}
        out['pooled'] = dict(
            n=int(len(err)), rmse=float(np.sqrt(np.mean(err ** 2))),
            spearman_sigma_err=float(stats.spearmanr(sig, err)[0]),
            spearman_dtr_err=float(stats.spearmanr(dtr, err)[0]),
            partial_err_sigma_given_dtr=partial_spearman(err, sig, dtr)[0],
            conformal_coverage_adaptive=float(cov.mean()),
            risk_coverage_rmse=rc)
        p = out['pooled']
        print(f"\nPOOLED FUTURE COMPOUNDS (n={p['n']}): RMSE {p['rmse']:.2f}  "
              f"sig->err {p['spearman_sigma_err']:+.3f}  partial|d {p['partial_err_sigma_given_dtr']:+.3f}  "
              f"90% adaptive coverage {p['conformal_coverage_adaptive']:.3f}")
        print(f"  risk-coverage: {rc['0.2']:.2f} (20%) -> {rc['1.0']:.2f} (all), "
              f"reduction {100*(rc['1.0']-rc['0.2'])/rc['1.0']:.0f}%")
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
