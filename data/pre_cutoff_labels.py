"""Rebuild each historical parent's activity from its pre-cutoff records only.

The temporal split assigns a compound by its FIRST disclosure, but the curated activity is the
median over ALL of that compound's records. A compound published before the cutoff and
re-measured after it therefore enters the historical pool carrying a label informed by
measurements that did not exist at the cutoff. That is look-ahead in the label.

Two ways of removing it are not equivalent. Dropping the affected compounds also removes them
from training, and they are not a random subset: they are measured five to seven times more
often than the compounds that stay, and are more potent, so dropping them selects the historical
sample on a post-cutoff event and changes what the model is fitted on. Recomputing the label
keeps every compound a modeller at the cutoff would have held, and gives each the value that
modeller would have computed. Only the label moves, and only for the affected compounds.

The archived curation keeps one aggregated value per parent and no per-record years, so the
per-record data are re-queried. A re-query is only trustworthy if it reproduces what was
archived, so every parent is validated against five independently stored fields before its label
is used: the record count, the first and last year, the document count and the aggregated
activity itself. A parent that fails any of them is reported rather than silently relabelled.

    python -m data.pre_cutoff_labels --cut 2015
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from data.recurate_chembl_v2 import (API, FIELDS, KEEP_TYPES, TARGETS, fetch_all,  # noqa: F401
                                     standard_parent)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'chembl_v2')


def parents_from_chembl(cid):
    """Re-run the archived curation filters and group records by standardized parent."""
    g = defaultdict(list)
    for a in fetch_all(cid):
        if a['standard_type'] not in KEEP_TYPES:
            continue
        if a['target_organism'] != 'Homo sapiens':
            continue
        try:
            v = float(a['standard_value'])
        except (TypeError, ValueError):
            continue
        if not (v > 0):
            continue
        _smi, key = standard_parent(a['canonical_smiles'])
        if key is None:
            continue
        g[key].append(dict(pAct=9.0 - np.log10(v), year=a['document_year'],
                           doc=a['document_chembl_id'], type=a['standard_type']))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', type=int, default=2015)
    ap.add_argument('--targets', nargs='+', default=['scd1', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--records-out', default=os.path.join(OUTD, 'temporal_records'))
    args = ap.parse_args()
    os.makedirs(args.records_out, exist_ok=True)
    report = {'cut': args.cut, 'targets': {}}

    for t in args.targets:
        src = os.path.join(OUTD, f'{t}_chembl_v2.csv')
        arch = {r['inchikey']: r for r in csv.DictReader(open(src))}
        live = parents_from_chembl(TARGETS[t])

        rows, stats = [], defaultdict(int)
        for key, r in arch.items():
            if not r.get('year_min'):
                stats['undated'] += 1
                continue
            recs = live.get(key)
            if not recs:
                stats['absent_on_requery'] += 1
                continue
            yrs = [int(x['year']) for x in recs if x['year']]
            vals = np.array([x['pAct'] for x in recs], float)
            # Five independently archived fields must all reproduce, or the record set has moved
            # and its pre-cutoff subset cannot be trusted.
            ok = (len(recs) == int(r['n_records'])
                  and yrs and int(min(yrs)) == int(r['year_min'])
                  and int(max(yrs)) == int(r['year_max'])
                  and len({x['doc'] for x in recs}) == int(r['n_documents'])
                  and abs(float(np.median(vals)) - float(r['pAct'])) < 1e-9)
            if not ok:
                stats['failed_validation'] += 1
                rows.append(dict(inchikey=key, validated=0, pAct_pre='', n_pre=0))
                continue
            stats['validated'] += 1
            pre = vals[[i for i, x in enumerate(recs) if x['year'] and int(x['year']) < args.cut]]
            spans = int(r['year_min']) < args.cut <= int(r['year_max'])
            stats['spanning'] += int(spans)
            rows.append(dict(inchikey=key, validated=1,
                             pAct_pre=(float(np.median(pre)) if len(pre) else ''),
                             n_pre=int(len(pre))))
            if spans and abs(float(np.median(pre)) - float(r['pAct'])) > 1e-9:
                stats['label_moved'] += 1
        with open(os.path.join(OUTD, f'{t}_pre_cutoff_labels.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, ['inchikey', 'validated', 'pAct_pre', 'n_pre'])
            w.writeheader(); w.writerows(rows)
        # Record-level provenance, archived so the historical labels can be rebuilt without a
        # second re-query. This is the file the curation never kept.
        with open(os.path.join(args.records_out, f'{t}_records.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['inchikey', 'year', 'standard_type', 'pAct'])
            for key in sorted(live):
                if key in arch:
                    for x in live[key]:
                        w.writerow([key, x['year'], x['type'], f"{x['pAct']:.6f}"])
        stats['records_undated'] = sum(1 for k in live if k in arch
                                       for x in live[k] if not x['year'])
        stats['parents_mixed_dates'] = sum(
            1 for k in live if k in arch
            and any(not x['year'] for x in live[k]) and any(x['year'] for x in live[k]))
        report['targets'][t] = dict(stats)
        print(f"  {t:6s} " + '  '.join(f'{k}={v}' for k, v in sorted(stats.items())))

    # A canonical, sorted manifest of every retained record, with its SHA-256. The five
    # parent-level validation fields show the aggregates reproduce; the manifest is what lets a
    # reader confirm the record set itself, which those aggregates alone could not establish.
    import hashlib
    man = os.path.join(args.records_out, 'MANIFEST.tsv')
    lines = []
    for t in sorted(args.targets):
        with open(os.path.join(args.records_out, f'{t}_records.csv')) as f:
            for row in sorted(list(csv.reader(f))[1:]):
                lines.append(t + '\t' + '\t'.join(row))
    body = '\n'.join(sorted(lines)) + '\n'
    open(man, 'w').write(body)
    report['manifest'] = {'path': os.path.relpath(man, os.path.dirname(HERE)),
                          'n_records': len(lines),
                          'sha256': hashlib.sha256(body.encode()).hexdigest()}
    print(f"  manifest: {len(lines)} records, sha256 {report['manifest']['sha256'][:16]}...")

    with open(os.path.join(OUTD, 'pre_cutoff_validation.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('wrote pre-cutoff labels, record-level provenance and the validation report')


if __name__ == '__main__':
    main()
