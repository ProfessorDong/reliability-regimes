"""Pseudo-prospective temporal validation (Tier-3).

Random cross-validation asks how a model behaves on compounds drawn from the same pool
it was fitted on. A prospective user is in a different position: the model is fitted on
what is known today and applied to chemistry that has not been published yet. The
re-curated ChEMBL files carry a publication year for every compound, so we can emulate
that directly.

Protocol, per target:
  * train on all compounds whose FIRST publication year is < CUT. First disclosure, not
    median, is the prospective quantity: a compound with one pre-CUT record and several
    post-CUT records was already public before the cutoff, and assigning it by median year
    would place known chemistry in the future set. Using year_min guarantees that no test
    compound carries any pre-CUT activity record.
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

    python -m reliability.run_temporal_v1 --cut 2015
"""
from __future__ import annotations
import argparse, csv, json, multiprocessing as mp, os, sys, time
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.run_applicability_v1 import max_tanimoto_to_set, partial_spearman  # noqa: E402
from reliability.oracle import featurize  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'chembl_v2')
OUT = os.path.join(BASE, 'outputs', 'frozen', 'temporal_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
YEAR_FIELD = 'year_min'      # set from --year-field; first disclosure by default
N_BOOT = 2000                # percentile bootstrap resamples for the temporal intervals
EPS = 1e-3


def _types(row):
    """The distinct ChEMBL standard types behind one standardized parent."""
    raw = row.get('types', '') or ''
    for sep in ('|', ',', ';'):
        raw = raw.replace(sep, ';')
    return {t.strip() for t in raw.split(';') if t.strip()}


def restrict_type(rows, cut, min_train=100, min_test=50):
    """Pick the single standard type that can carry a temporal test on this target.

    Not simply the commonest type. NK1R is mostly IC50 overall, but its post-cutoff records are
    mostly Ki, so an IC50-only split leaves 47 future compounds and cannot be evaluated. That
    turnover is the very thing this analysis exists to probe, so the rule selects on the binding
    constraint instead: among types whose pure-type parents give a usable split, take the one
    with the most post-cutoff compounds. Returns None when no type qualifies.
    """
    best, best_n = None, -1
    types = {t for r in rows for t in _types(r) if len(_types(r)) == 1}
    for ty in sorted(types):
        sub = [r for r in rows if _types(r) == {ty}]
        yrs = [int(r[YEAR_FIELD]) for r in sub]
        ntr = sum(1 for v in yrs if v < cut)
        nte = sum(1 for v in yrs if v >= cut)
        if ntr >= min_train and nte >= min_test and nte > best_n:
            best, best_n = ty, nte
    return best


def load(target, endpoint=None, cut=2015, exclude_spanning=False, pre_cutoff_labels=False):
    """Rows for one target, optionally restricted to a single measurement type.

    `endpoint='dominant'` keeps only parents whose retained ChEMBL records are ALL of the
    target's most common standard type. That drops both the mixed-type parents, whose median
    aggregates across biologically different quantities, and the minority-type ones, so the
    label becomes one kind of measurement rather than a pooled scale. It is the direct test of
    whether the temporal degradation is a chemical effect or a change of measurement regime
    across the cutoff, which the endpoint-turnover analysis can only address indirectly.
    """
    rows = list(csv.DictReader(open(os.path.join(DATA, f'{target}_chembl_v2.csv'))))
    rows = [r for r in rows if r.get(YEAR_FIELD)]
    if exclude_spanning:
        # A parent's activity is the median over ALL of its records, while the split is decided
        # by its FIRST disclosure. A compound first published before the cutoff but re-measured
        # after it therefore enters the historical pool carrying a label that absorbs later
        # measurements. Those parents have no well-defined historical label from the archived
        # data, which keeps no per-record years, so this drops them rather than guessing.
        rows = [r for r in rows
                if not (int(r[YEAR_FIELD]) < cut and r.get('year_max')
                        and int(r['year_max']) >= cut)]
    kept_type = None
    if endpoint == 'single':
        kept_type = restrict_type(rows, cut)
        rows = [r for r in rows if _types(r) == {kept_type}] if kept_type else []
    smi = [r['SMILES'] for r in rows]
    y = np.array([float(r['pAct']) for r in rows])
    yr = np.array([int(r[YEAR_FIELD]) for r in rows])
    if pre_cutoff_labels:
        # Replace each historical compound's activity with the median of its records published
        # before the cutoff. A compound first disclosed before the cutoff but re-measured after it
        # would otherwise carry a label that no one could have computed at the cutoff. Evaluation
        # compounds are untouched: their first disclosure is at or after the cutoff, so every
        # record they carry is already post-cutoff.
        lab = {r['inchikey']: r for r in csv.DictReader(
            open(os.path.join(DATA, f'{target}_pre_cutoff_labels.csv')))}
        keys = [r['inchikey'] for r in rows]
        miss = [k for k, v in zip(keys, yr) if v < cut and lab.get(k, {}).get('validated') != '1']
        if miss:
            raise SystemExit(f'{target}: {len(miss)} historical parents lack a validated '
                             f'pre-cutoff label; refusing to relabel from unverified data')
        for i, (k, v) in enumerate(zip(keys, yr)):
            if v < cut:
                y[i] = float(lab[k]['pAct_pre'])
    return (smi, y, yr, kept_type) if endpoint else (smi, y, yr)


def conformal_q(scores, alpha=0.1):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else float(np.max(scores))


# --- size-matched control, run across processes --------------------------------------------
# The replicates are independent and each is seeded from its own index, so a replicate's value
# does not depend on how many are run or in what order. Workers are forked after the feature
# matrix is built, which shares it copy-on-write, and each fit is given one core: a forest is
# bit-identical across n_jobs, so this changes only the wall clock.
_W = {}


def _control_rep(rep):
    X, y = _W['X'], _W['y']
    nte, ncal, nptr = _W['nte'], _W['ncal'], _W['nptr']
    r2 = np.random.default_rng(100 + rep)
    pe = r2.permutation(len(y))
    c_te = pe[:nte]; c_cal = pe[nte:nte + ncal]; c_ptr = pe[nte + ncal:nte + ncal + nptr]
    kw = dict(RF_KW); kw['n_jobs'] = 1
    rfc = RandomForestRegressor(**kw).fit(X[c_ptr], y[c_ptr])

    def pc(idx):
        Q = np.stack([t.predict(X[idx]) for t in rfc.estimators_], 0)
        return Q.mean(0), Q.std(0)

    yc2, sc2 = pc(c_cal); yt2, st2 = pc(c_te)
    e_c = np.abs(y[c_cal] - yc2); e_t = np.abs(y[c_te] - yt2)
    qa2 = conformal_q(e_c / (sc2 + EPS), 0.1)
    return (float(stats.spearmanr(st2, e_t)[0]),
            float(np.sqrt(np.mean(e_t ** 2))),
            float(np.mean(e_t <= qa2 * (st2 + EPS))))


def emp_p(ctl, obs, worse):
    """One-sided empirical P against the control distribution.

    The fraction of control replicates at least as extreme as the single temporal value, with
    the conventional +1 in numerator and denominator so the value can never be zero. The
    smallest attainable value is 1/(R+1), which is why R matters: at R=20 nothing below 0.048
    can be reported however extreme the observation is.
    """
    v = np.asarray(ctl, float)
    k = int((v >= obs).sum()) if worse == 'greater' else int((v <= obs).sum())
    return float((1 + k) / (len(v) + 1))


def delta_ci(obs, boot, ctl, rng):
    """Interval for the direct effect, temporal minus the control mean.

    Whether two separately constructed intervals overlap is not itself an interval for their
    difference, so the two sources are combined instead. The temporal arm contributes its
    bootstrap over evaluation compounds, which does not shrink with more control replicates;
    the control mean contributes the standard error of a mean over R independent replicates,
    which does. Drawing from both preserves any skew in the temporal bootstrap.
    """
    b = np.asarray(boot, float); v = np.asarray(ctl, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    d = (b - b.mean() + obs) - (v.mean() + se * rng.standard_normal(len(b)))
    return dict(delta=float(obs - v.mean()),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                control_mean_se=float(se))


def scaffold_ids(smiles):
    """Bemis-Murcko scaffold per structure, for a bootstrap that resamples series."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    out = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else f'_{i}')
        except Exception:
            out.append(f'_{i}')
    return np.array([x if x else f'_{i}' for i, x in enumerate(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--cut', type=int, default=2015)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--control-reps', type=int, default=1000,
                    help='random size-matched control replicates. The control interval narrows '
                         'as this grows, and the smallest reportable empirical P is 1/(R+1), '
                         'so R also sets the resolution of the comparison')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help='processes for the control replicates; each fit uses one core, and a '
                         'forest is bit-identical across n_jobs, so this changes only run time')
    ap.add_argument('--pre-cutoff-labels', action='store_true',
                    help='rebuild each historical parent activity from its pre-cutoff records '
                         'only, keeping every compound. Off by default so the frozen analysis '
                         'path is unchanged.')
    ap.add_argument('--exclude-spanning', action='store_true',
                    help='drop pre-cutoff parents whose records extend past the cutoff, whose '
                         'median label would otherwise absorb post-cutoff measurements. Off by '
                         'default so the frozen analysis path is unchanged.')
    ap.add_argument('--endpoint', default=None, choices=['single'],
                    help='restrict each target to parents whose records are all of its most '
                         'common standard type. Off by default, so the frozen analysis path is '
                         'unchanged; on, it tests whether the temporal effect survives when the '
                         'label is one measurement type rather than a pooled scale')
    ap.add_argument('--year-field', default='year_min', choices=['year_min', 'year_median'],
                    help='year_min is first disclosure and is the prospective default; '
                         'year_median is retained only as a sensitivity analysis')
    args = ap.parse_args()
    global YEAR_FIELD
    YEAR_FIELD = args.year_field
    out = {'cut_year': args.cut,
           'year_field': args.year_field,
           'protocol': 'train on compounds first published before CUT, evaluate on those first '
                       'published at CUT or later; grouping on standardized parent InChIKey; '
                       'calibration uses pre-CUT compounds only'}
    P = {k: [] for k in ('err', 'sig', 'dtr', 'cov')}
    out['endpoint_restriction'] = args.endpoint or 'none'
    out['exclude_spanning'] = bool(args.exclude_spanning)
    out['pre_cutoff_labels'] = bool(args.pre_cutoff_labels)
    for tgt in args.targets:
        if args.endpoint:
            smi, y, yr, kept = load(tgt, args.endpoint, args.cut, args.exclude_spanning, args.pre_cutoff_labels)
            out.setdefault('kept_type', {})[tgt] = kept
        else:
            smi, y, yr = load(tgt, None, args.cut, args.exclude_spanning, args.pre_cutoff_labels)
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
        # The temporal split is a single realisation, so its uncertainty is the uncertainty of
        # the evaluation set. One percentile bootstrap over the test compounds gives the error,
        # the coverage and the ranking their intervals from the same resamples. The conformal
        # quantile is held fixed, since it is calibrated on past compounds and is not itself
        # being resampled.
        _bs = np.random.default_rng(7)
        _br, _bc, _bq = [], [], []
        for _ in range(N_BOOT):
            i = _bs.integers(0, len(err_t), len(err_t))
            _br.append(np.sqrt(np.mean(err_t[i] ** 2)))
            _bc.append(cov_ada[i].mean())
            _bq.append(stats.spearmanr(st_[i], err_t[i])[0])
        pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        rho_ci, rmse_ci, cov_ci = pct(_bq), pct(_br), pct(_bc)
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
            rmse_test_ci95=rmse_ci,
            spearman_sigma_err=float(rho_sig[0]), p_sigma_err=float(rho_sig[1]),
            spearman_sigma_err_ci95=rho_ci,
            spearman_dtr_err=float(rho_dtr[0]),
            partial_err_sigma_given_dtr=part[0],
            conformal_coverage_standard=float(cov_std.mean()),
            conformal_coverage_adaptive=float(cov_ada.mean()),
            conformal_coverage_adaptive_ci95=cov_ci,
            conformal_width_standard=float(2 * q_std),
            conformal_width_adaptive_median=float(np.median(2 * q_ada * (st_ + EPS))),
            risk_coverage_rmse=rc,
        )
        # SIZE-MATCHED RANDOM CONTROL: same n_train / n_cal / n_test, drawn at random.
        # If the temporal degradation were only a sample-size effect, this control would
        # degrade equally. Unlike the temporal split, which happens once, the control is a
        # distribution over random splits, so its interval is taken across replicates rather
        # than by bootstrap: a t interval on the mean, and the replicate range, which is what
        # the single temporal value should be judged extreme against.
        _W.update(X=X, y=y, nte=len(te), ncal=len(cal), nptr=len(ptr))
        _t0 = time.time()
        if args.jobs == 1:
            _res = [_control_rep(r) for r in range(args.control_reps)]
        else:
            with mp.get_context('fork').Pool(args.jobs) as _pool:
                _res = _pool.map(_control_rep, range(args.control_reps), chunksize=1)
        ctl = {'rho': [r[0] for r in _res], 'rmse': [r[1] for r in _res],
               'cov': [r[2] for r in _res]}
        print(f"        {args.control_reps} control replicates in {time.time() - _t0:.0f}s "
              f"on {args.jobs} processes", flush=True)
        nrep = args.control_reps
        tcrit = float(stats.t.ppf(0.975, nrep - 1))

        def ctl_stats(v):
            v = np.asarray(v, float)
            m, sd = float(v.mean()), float(v.std(ddof=1))
            return m, sd, [m - tcrit * sd / np.sqrt(nrep), m + tcrit * sd / np.sqrt(nrep)], \
                [float(v.min()), float(v.max())]

        c_rho, c_rmse, c_cov = ctl_stats(ctl['rho']), ctl_stats(ctl['rmse']), ctl_stats(ctl['cov'])
        out[tgt]['control_random_same_size'] = dict(
            spearman_sigma_err=c_rho[0], rmse=c_rmse[0], conformal_coverage_adaptive=c_cov[0],
            n_reps=nrep,
            spearman_sigma_err_sd=c_rho[1], rmse_sd=c_rmse[1],
            conformal_coverage_adaptive_sd=c_cov[1],
            spearman_sigma_err_ci95=c_rho[2], rmse_ci95=c_rmse[2],
            conformal_coverage_adaptive_ci95=c_cov[2],
            spearman_sigma_err_range=c_rho[3], rmse_range=c_rmse[3],
            conformal_coverage_adaptive_range=c_cov[3],
            # whether the single temporal value lies outside the whole replicate range
            temporal_outside_range=dict(
                rmse=bool(out[tgt]['rmse_test'] > c_rmse[3][1]),
                coverage=bool(out[tgt]['conformal_coverage_adaptive'] < c_cov[3][0]),
                spearman=bool(out[tgt]['spearman_sigma_err'] < c_rho[3][0])),
            # Exact one-sided empirical P against the control distribution, in the direction
            # that counts as degradation for each measure: error higher, coverage lower,
            # ranking lower.
            empirical_p=dict(
                rmse=emp_p(ctl['rmse'], out[tgt]['rmse_test'], 'greater'),
                coverage=emp_p(ctl['cov'], out[tgt]['conformal_coverage_adaptive'], 'less'),
                spearman=emp_p(ctl['rho'], out[tgt]['spearman_sigma_err'], 'less'),
                floor=1.0 / (nrep + 1)))
        # Direct effect against the control mean, with the two sources of uncertainty combined
        # rather than read off the overlap of two separate intervals.
        _dr = np.random.default_rng(21)
        out[tgt]['delta_vs_control'] = dict(
            rmse=delta_ci(out[tgt]['rmse_test'], _br, ctl['rmse'], _dr),
            spearman=delta_ci(out[tgt]['spearman_sigma_err'], _bq, ctl['rho'], _dr),
            coverage=delta_ci(out[tgt]['conformal_coverage_adaptive'], _bc, ctl['cov'], _dr))
        # Sensitivity: compounds from one chemical series are not independent, so the temporal
        # interval is recomputed resampling Bemis-Murcko scaffold groups rather than compounds.
        _sc = scaffold_ids([smi[i] for i in te])
        _grp = {}
        for _i, _s in enumerate(_sc):
            _grp.setdefault(_s, []).append(_i)
        _keys = list(_grp)
        _cb = np.random.default_rng(31)
        _cr, _cc, _cq = [], [], []
        for _ in range(N_BOOT):
            _pick = [_grp[_keys[j]] for j in _cb.integers(0, len(_keys), len(_keys))]
            i = np.fromiter((x for g in _pick for x in g), int)
            _cr.append(np.sqrt(np.mean(err_t[i] ** 2)))
            _cc.append(cov_ada[i].mean())
            _cq.append(stats.spearmanr(st_[i], err_t[i])[0])
        out[tgt]['scaffold_cluster_bootstrap'] = dict(
            n_scaffolds=len(_keys),
            rmse_ci95=pct(_cr), conformal_coverage_adaptive_ci95=pct(_cc),
            spearman_sigma_err_ci95=[float(np.nanpercentile(_cq, 2.5)),
                                     float(np.nanpercentile(_cq, 97.5))])
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
        print(f"        95% CI: RMSE [{o['rmse_test_ci95'][0]:.2f}, {o['rmse_test_ci95'][1]:.2f}]  "
              f"sig->err [{o['spearman_sigma_err_ci95'][0]:+.3f}, {o['spearman_sigma_err_ci95'][1]:+.3f}]  "
              f"coverage [{o['conformal_coverage_adaptive_ci95'][0]:.3f}, "
              f"{o['conformal_coverage_adaptive_ci95'][1]:.3f}]", flush=True)
        c = out[tgt]['control_random_same_size']
        print(f"        size-matched RANDOM control (n={c['n_reps']}): "
              f"RMSE {c['rmse']:.2f} [{c['rmse_ci95'][0]:.2f}, {c['rmse_ci95'][1]:.2f}]  "
              f"sig->err {c['spearman_sigma_err']:+.3f} "
              f"[{c['spearman_sigma_err_ci95'][0]:+.3f}, {c['spearman_sigma_err_ci95'][1]:+.3f}]  "
              f"coverage {c['conformal_coverage_adaptive']:.3f} "
              f"[{c['conformal_coverage_adaptive_ci95'][0]:.3f}, "
              f"{c['conformal_coverage_adaptive_ci95'][1]:.3f}]", flush=True)
        _o = c['temporal_outside_range']
        print(f"        temporal outside the full control replicate range?  "
              f"RMSE {_o['rmse']}  coverage {_o['coverage']}  ranking {_o['spearman']}", flush=True)
    if P['err']:
        err = np.concatenate(P['err']); sig = np.concatenate(P['sig'])
        dtr = np.concatenate(P['dtr']); cov = np.concatenate(P['cov'])
        order = np.argsort(sig); e = err[order]
        rc = {f'{c:.1f}': float(np.sqrt(np.mean(e[:max(1, int(c * len(e)))] ** 2)))
              for c in (0.2, 0.4, 0.6, 0.8, 1.0)}
        _bs = np.random.default_rng(11)
        _pr, _pc, _pq = [], [], []
        for _ in range(N_BOOT):
            i = _bs.integers(0, len(err), len(err))
            _pr.append(np.sqrt(np.mean(err[i] ** 2)))
            _pc.append(cov[i].mean())
            _pq.append(stats.spearmanr(sig[i], err[i])[0])
        pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out['pooled'] = dict(
            n=int(len(err)), rmse=float(np.sqrt(np.mean(err ** 2))),
            rmse_ci95=pct(_pr), conformal_coverage_adaptive_ci95=pct(_pc),
            spearman_sigma_err_ci95=pct(_pq),
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
    # Select the per-target entries by their shape rather than by excluding the names of the
    # non-target keys. An exclusion list silently admits any key added later: 'kept_type' and
    # 'endpoint_restriction' were both swept in this way, and a string cannot be subscripted
    # with 'n_test'. Only a per-target result carries n_test, so that is the test.
    _t = [k for k, v in out.items() if isinstance(v, dict) and 'n_test' in v]
    if _t:
        _n = np.array([out[k]['n_test'] for k in _t], float)
        _mean = lambda f: float(np.mean([f(out[k]) for k in _t]))
        _wr = lambda f: float(np.sqrt(np.sum(_n * np.array([f(out[k]) for k in _t]) ** 2) / _n.sum()))
        out['macro'] = dict(
            n_targets=len(_t), targets=_t,
            temporal=dict(rmse=_mean(lambda d: d['rmse_test']),
                          rmse_pooled_nweighted=_wr(lambda d: d['rmse_test']),
                          spearman_sigma_err=_mean(lambda d: d['spearman_sigma_err']),
                          conformal_coverage_adaptive=_mean(lambda d: d['conformal_coverage_adaptive'])),
            control=dict(rmse=_mean(lambda d: d['control_random_same_size']['rmse']),
                         rmse_pooled_nweighted=_wr(lambda d: d['control_random_same_size']['rmse']),
                         spearman_sigma_err=_mean(lambda d: d['control_random_same_size']['spearman_sigma_err']),
                         conformal_coverage_adaptive=_mean(
                             lambda d: d['control_random_same_size']['conformal_coverage_adaptive'])))
        m = out['macro']
        # the size-matched control is the only like-for-like comparator: same data, same sizes
        out['delta_vs_control'] = {
            k: {'temporal': m['temporal'][k], 'control': m['control'][k],
                'delta': m['temporal'][k] - m['control'][k]}
            for k in ('rmse', 'spearman_sigma_err', 'conformal_coverage_adaptive')}
        out['macro']['temporal_outside_control_range'] = {
            k: sum(out[t]['control_random_same_size']['temporal_outside_range'][k] for t in _t)
            for k in ('rmse', 'coverage', 'spearman')}
        out['delta_vs_control']['rmse_pct_increase_vs_control'] = 100.0 * (
            m['temporal']['rmse_pooled_nweighted'] / m['control']['rmse_pooled_nweighted'] - 1.0)
        out['per_target_delta'] = {k: {
            'd_rmse': out[k]['rmse_test'] - out[k]['control_random_same_size']['rmse'],
            'd_rho': out[k]['spearman_sigma_err'] - out[k]['control_random_same_size']['spearman_sigma_err'],
            'd_cov': out[k]['conformal_coverage_adaptive']
                     - out[k]['control_random_same_size']['conformal_coverage_adaptive']} for k in _t}
        print('\nMACRO (equal weight over targets), temporal vs its size-matched control:')
        for k, v in out['delta_vs_control'].items():
            if isinstance(v, dict):
                print(f"  {k:28s} {v['temporal']:+.3f} vs {v['control']:+.3f}   delta {v['delta']:+.3f}")
        print(f"  pooled n-weighted RMSE   {m['temporal']['rmse_pooled_nweighted']:.3f} vs "
              f"{m['control']['rmse_pooled_nweighted']:.3f}  "
              f"(+{out['delta_vs_control']['rmse_pct_increase_vs_control']:.0f}%)")
        print('  per-target delta vs control:')
        for k, v in out['per_target_delta'].items():
            print(f"    {k:6s} dRMSE {v['d_rmse']:+.3f}  drho {v['d_rho']:+.3f}  dcov {v['d_cov']:+.3f}")

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
