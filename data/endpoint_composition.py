#!/usr/bin/env python3
"""Document which activity endpoints each curated dataset actually pools.

The manuscript reports every activity on one -log10 molar scale. That scale is
populated from more than one ChEMBL standard_type, so the composition has to be
stated rather than assumed. This script measures it in two places.

The temporal cohort is measured directly: data/chembl_v2/*.csv retain a `types`
field per parent structure, so the composition needs no network access. It is
also split at the temporal cutoff, which tests whether endpoint composition
itself shifts across the split and could stand in for the chemical shift the
temporal analysis attributes the error rise to.

The cross-validation cohort keeps only SMILES and activity, so its composition
is recovered by matching its structures back to ChEMBL. SCD-1 and FADS are
mostly literature-panel compounds and match only in part; the matched fraction
is reported so the coverage of the statement is visible.

Writes outputs/frozen/endpoint_composition.json.
"""
from __future__ import annotations
import collections
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'outputs', 'frozen', 'endpoint_composition.json')

API = 'https://www.ebi.ac.uk/chembl/api/data/activity'
KEEP_TYPES = ('IC50', 'Ki', 'Kd', 'EC50')
CUT = 2015

# ChEMBL target identifiers, matching data/recurate_chembl_v2.py.
CHEMBL_TARGETS = {'scd1': 'CHEMBL5555', 'nk1r': 'CHEMBL249',
                  'drd2': 'CHEMBL217', 'drd3': 'CHEMBL234'}

CV_FILES = {'scd1': 'scd1_binding.csv',
            'fads': 'fatty_acid_desaturase_bioactivity.csv',
            'nk1r': 'nk1r_combined.csv',
            'drd2': 'drd2_bioactivity.csv',
            'drd3': 'drd3_chembl.csv'}


def _pct(counter):
    n = sum(counter.values())
    return {k: round(100.0 * v / n, 1) for k, v in counter.most_common()} if n else {}


def temporal_composition():
    """Endpoint mix of the re-curated temporal cohort, and its shift at the cutoff."""
    out = {}
    for t in sorted(CHEMBL_TARGETS):
        path = os.path.join(HERE, 'chembl_v2', '%s_chembl_v2.csv' % t)
        rows = list(csv.DictReader(open(path)))
        overall, pre, post = collections.Counter(), collections.Counter(), collections.Counter()
        n_pre = n_post = n_undated = 0
        for r in rows:
            types = [x for x in r['types'].split(';') if x]
            overall.update(types)
            y = r['year_min'].strip()
            if not y:
                n_undated += 1
                continue
            if int(float(y)) < CUT:
                pre.update(types); n_pre += 1
            else:
                post.update(types); n_post += 1
        out[t] = {'n_parents': len(rows), 'n_undated': n_undated,
                  'overall_pct': _pct(overall),
                  'n_pre': n_pre, 'pre_pct': _pct(pre),
                  'n_post': n_post, 'post_pct': _pct(post),
                  # Total variation distance between the pre- and post-cutoff
                  # endpoint distributions: 0 means composition is unchanged.
                  'tv_distance': round(0.5 * sum(
                      abs(_pct(pre).get(k, 0.0) - _pct(post).get(k, 0.0))
                      for k in set(_pct(pre)) | set(_pct(post))) / 100.0, 3)}
    pooled = collections.Counter()
    for t in out:
        path = os.path.join(HERE, 'chembl_v2', '%s_chembl_v2.csv' % t)
        for r in csv.DictReader(open(path)):
            pooled.update([x for x in r['types'].split(';') if x])
    out['pooled_pct'] = _pct(pooled)
    return out


def _fetch(target_id):
    """Pull every exact, nanomolar, affinity record for one ChEMBL target."""
    got, offset = [], 0
    while True:
        q = ('%s?target_chembl_id=%s&standard_relation=%s&standard_units=nM'
             '&limit=1000&offset=%d&format=json'
             % (API, target_id, urllib.parse.quote('='), offset))
        with urllib.request.urlopen(q, timeout=120) as fh:
            page = json.load(fh)
        acts = page['activities']
        got.extend(acts)
        if len(acts) < 1000:
            break
        offset += 1000
        time.sleep(0.2)
    return got


def cv_composition():
    """Endpoint mix of the cross-validation cohort, recovered by structure match."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    def canon(s):
        m = Chem.MolFromSmiles(s) if s else None
        return Chem.MolToSmiles(m) if m is not None else None

    out = {}
    for t, fname in sorted(CV_FILES.items()):
        rows = list(csv.DictReader(open(os.path.join(HERE, fname))))
        want = {c for c in (canon(r['SMILES']) for r in rows) if c}
        if t not in CHEMBL_TARGETS:
            # FADS holds two ChEMBL records in total; the set is a literature panel.
            out[t] = {'n_records': len(rows), 'n_structures': len(want),
                      'matched': 0, 'matched_pct': 0.0, 'types_pct': {},
                      'note': 'literature panel; not recoverable from ChEMBL'}
            continue
        sys.stderr.write('  fetching %s (%s) ...\n' % (t, CHEMBL_TARGETS[t]))
        acts = _fetch(CHEMBL_TARGETS[t])
        types = collections.Counter()
        seen = set()
        for a in acts:
            if a.get('standard_type') not in KEEP_TYPES:
                continue
            c = canon(a.get('canonical_smiles'))
            if c is None or c not in want:
                continue
            types[a['standard_type']] += 1
            seen.add(c)
        out[t] = {'n_records': len(rows), 'n_structures': len(want),
                  'n_chembl_records_pulled': len(acts),
                  'matched': len(seen),
                  'matched_pct': round(100.0 * len(seen) / len(want), 1) if want else 0.0,
                  'types_pct': _pct(types), 'n_matched_records': sum(types.values())}
    return out


def main():
    res = {'cutoff': CUT, 'keep_types': list(KEEP_TYPES),
           'temporal_cohort': temporal_composition()}
    sys.stderr.write('temporal cohort done; querying ChEMBL for the CV cohort\n')
    res['cv_cohort'] = cv_composition()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
