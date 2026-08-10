"""Compare the temporal analysis under a single-measurement-type label against the pooled one.

The pooled label mixes IC50, Ki, Kd and EC50 on one negative-log molar scale. If the endpoint
mix turns over across the 2015 cutoff, part of what reads as temporal degradation could be a
change in what is being measured rather than a change in what generalizes. This script sets the
two analyses side by side: the frozen pooled run, and a re-run in which each target keeps only
parents whose records are all of one standard type.

It performs no fits. It reads two frozen outputs and writes a comparison JSON.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, '..', 'outputs', 'frozen')

TARGETS = ['scd1', 'nk1r', 'drd2', 'drd3']

# short name, key in the temporal arm, key in delta_vs_control, direction of degradation
MEASURES = [
    ('rmse', 'rmse_test', 'rmse', +1),
    ('rho', 'spearman_sigma_err', 'spearman', -1),
    ('cov', 'conformal_coverage_adaptive', 'coverage', -1),
]


def per_target(run, tgt):
    """Temporal value, control mean, and the difference with its interval, for one target."""
    r = run[tgt]
    ctl = r['control_random_same_size']
    dvc = r['delta_vs_control']
    out = {'n_train': r['n_train'], 'n_test': r['n_test'],
           'empirical_p': ctl['empirical_p']}
    for short, tkey, dkey, _ in MEASURES:
        out[short] = {
            'temporal': r[tkey],
            'control': ctl[tkey if tkey != 'rmse_test' else 'rmse'],
            'delta': dvc[dkey]['delta'],
            'delta_ci95': dvc[dkey]['ci95'],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pooled', default=os.path.join(FROZEN, 'temporal_analysis.json'))
    ap.add_argument('--restricted', default=os.path.join(FROZEN, 'temporal_endpoint.json'))
    ap.add_argument('--out', default=os.path.join(FROZEN, 'temporal_endpoint_comparison.json'))
    args = ap.parse_args()

    pool = json.load(open(args.pooled))
    rest = json.load(open(args.restricted))

    assert pool.get('endpoint_restriction', 'none') == 'none', 'the pooled run is itself restricted'
    assert rest.get('endpoint_restriction') == 'single', 'the restricted run carries no restriction'
    assert pool['cut_year'] == rest['cut_year'], 'the two runs use different cutoffs'
    assert pool['year_field'] == rest['year_field'], 'the two runs date compounds differently'

    cmp = {'cut_year': pool['cut_year'], 'year_field': pool['year_field'],
           'control_reps': rest['scd1']['control_random_same_size']['n_reps'],
           'kept_type': rest['kept_type'], 'per_target': {}}

    for t in TARGETS:
        p, r = per_target(pool, t), per_target(rest, t)
        cmp['per_target'][t] = {
            'kept_type': rest['kept_type'][t],
            'pooled': p, 'restricted': r,
            'retained_frac_train': r['n_train'] / p['n_train'],
            'retained_frac_test': r['n_test'] / p['n_test'],
            # A degradation is reproduced when the difference from the control keeps its
            # sign, whatever the size of the effect on the smaller single-type subset.
            'sign_agrees': {s: (p[s]['delta'] > 0) == (r[s]['delta'] > 0)
                            for s, _, _, _ in MEASURES},
            'restricted_degrades': {s: sgn * r[s]['delta'] > 0
                                    for s, _, _, sgn in MEASURES},
        }

    # Macro effect, equally weighted across targets, in each analysis.
    for label in ('pooled', 'restricted'):
        cmp.setdefault('macro_delta', {})[label] = {
            s: sum(cmp['per_target'][t][label][s]['delta'] for t in TARGETS) / len(TARGETS)
            for s, _, _, _ in MEASURES}

    cmp['conclusion'] = {
        'rmse_degrades_on_all_targets':
            all(cmp['per_target'][t]['restricted_degrades']['rmse'] for t in TARGETS),
        'rmse_p_at_floor_n': sum(
            cmp['per_target'][t]['restricted']['empirical_p']['rmse']
            <= cmp['per_target'][t]['restricted']['empirical_p']['floor'] for t in TARGETS),
        'sign_agrees_n': {s: sum(cmp['per_target'][t]['sign_agrees'][s] for t in TARGETS)
                          for s, _, _, _ in MEASURES},
        'macro_rmse_effect_pooled': cmp['macro_delta']['pooled']['rmse'],
        'macro_rmse_effect_restricted': cmp['macro_delta']['restricted']['rmse'],
    }

    json.dump(cmp, open(args.out, 'w'), indent=1)

    print(f"cutoff {cmp['cut_year']} by {cmp['year_field']}, "
          f"{cmp['control_reps']} control replicates\n")
    print(f"{'target':6s} {'type':5s} {'kept tr/te':>11s}   "
          f"{'RMSE pooled':>19s} {'RMSE 1-type':>19s}   "
          f"{'rho pooled':>19s} {'rho 1-type':>19s}   "
          f"{'cov pooled':>19s} {'cov 1-type':>19s}")
    for t in TARGETS:
        c = cmp['per_target'][t]
        line = (f"{t:6s} {c['kept_type'] or '-':5s} "
                f"{c['retained_frac_train']:4.0%}/{c['retained_frac_test']:4.0%}   ")
        for s, _, _, _ in MEASURES:
            for lab in ('pooled', 'restricted'):
                m = c[lab][s]
                line += f"{m['temporal']:6.3f} v {m['control']:6.3f}  "
            line += "  "
        print(line)
    print("\nmacro difference from the size-matched control:")
    for lab in ('pooled', 'restricted'):
        d = cmp['macro_delta'][lab]
        print(f"  {lab:11s} RMSE {d['rmse']:+.3f}   ranking {d['rho']:+.3f}   "
              f"coverage {d['cov']:+.3f}")
    print("\nper-target one-sided Monte Carlo P for the RMSE rise (restricted):")
    for t in TARGETS:
        ep = cmp['per_target'][t]['restricted']['empirical_p']
        print(f"  {t:6s} {ep['rmse']:.4f}   (floor {ep['floor']:.4f})")
    print(f"\n{json.dumps(cmp['conclusion'], indent=1)}")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
