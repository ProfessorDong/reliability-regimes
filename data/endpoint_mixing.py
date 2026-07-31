#!/usr/bin/env python3
"""How often does median aggregation pool different endpoint types within one compound?

Curation collapses records sharing a standardized parent InChIKey to their median activity.
The retained records are not all the same measurement: they carry IC50, Ki, Kd and EC50 on one
negative-logarithmic molar axis. Where a single parent carries more than one of those types,
the median is taken across biologically different quantities rather than across replicates of
one, which is a stronger assumption than aggregating within a type.

The Methods disclose the endpoint composition per target (Table S20) but not how often the
aggregation step itself crosses types, and that is the part the assumption rests on. This
script measures it, and measures the consequence: the within-parent dispersion of parents whose
records share one type against those that mix.

Reads the re-curated ChEMBL files, which retain a per-parent `types` field and the within-parent
standard deviation. Writes outputs/frozen/endpoint_mixing.json.

    python -m data.endpoint_mixing
"""
from __future__ import annotations
import csv
import glob
import json
import os
import statistics as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'data', 'chembl_v2')
OUT = os.path.join(BASE, 'outputs', 'frozen', 'endpoint_mixing.json')


def _types(row):
    """The distinct ChEMBL standard types behind one standardized parent."""
    raw = row.get('types', '') or ''
    for sep in ('|', ',', ';'):
        raw = raw.replace(sep, ';')
    return {t.strip() for t in raw.split(';') if t.strip()}


def _sd(rows):
    v = [float(r['sd']) for r in rows
         if (r.get('sd') or '').strip() not in ('', 'nan', 'None')]
    return float(st.mean(v)) if v else None


def main():
    res = {}
    n_all = n_mixed = 0
    for path in sorted(glob.glob(os.path.join(SRC, '*_chembl_v2.csv'))):
        target = os.path.basename(path).split('_')[0]
        rows = list(csv.DictReader(open(path)))
        multi = [r for r in rows if int(r['n_records']) > 1]
        mixed = [r for r in multi if len(_types(r)) > 1]
        single = [r for r in multi if len(_types(r)) == 1]
        res[target] = dict(
            n_parents=len(rows),
            n_multi_record=len(multi),
            n_mixed_type=len(mixed),
            pct_mixed_of_parents=100.0 * len(mixed) / len(rows) if rows else 0.0,
            sd_single_type=_sd(single),
            sd_mixed_type=_sd(mixed),
        )
        n_all += len(rows)
        n_mixed += len(mixed)
        r = res[target]
        print('  %-6s %5d parents, %4d multi-record, %4d mix types (%.1f%%); '
              'within-parent SD %s single vs %s mixed'
              % (target, r['n_parents'], r['n_multi_record'], r['n_mixed_type'],
                 r['pct_mixed_of_parents'],
                 '%.2f' % r['sd_single_type'] if r['sd_single_type'] is not None else '-',
                 '%.2f' % r['sd_mixed_type'] if r['sd_mixed_type'] is not None else '-'))
    res['pooled'] = dict(n_parents=n_all, n_mixed_type=n_mixed,
                         pct_mixed_of_parents=100.0 * n_mixed / n_all if n_all else 0.0)
    # The targets where mixing inflates within-parent dispersion, which is the measurable
    # cost of crossing types in the median.
    res['targets_sd_inflated'] = sorted(
        t for t, v in res.items()
        if isinstance(v, dict) and v.get('sd_mixed_type') and v.get('sd_single_type')
        and v['sd_mixed_type'] > v['sd_single_type'])
    print('\n  pooled: %d of %d parents mix endpoint types (%.1f%%); dispersion inflated on %s'
          % (n_mixed, n_all, res['pooled']['pct_mixed_of_parents'],
             ', '.join(res['targets_sd_inflated']) or 'none'))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
