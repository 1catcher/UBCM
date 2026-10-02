# -*- coding: utf-8 -*-
"""Paired McNemar tests for RULER AR: ubcm vs retrieval / truncation."""
import json
from collections import defaultdict

from scipy.stats import binomtest

c = json.load(open('mab_cache.json', encoding='utf-8'))
rows = defaultdict(dict)


def parse(k):
    if k.startswith('AR|'):
        p = k.split('|')
        return int(p[1][1:]), int(p[2][1:]), p[3], int(p[4])
    v = json.loads(k)
    return int(v[1]), int(v[2]), v[3], int(v[4])


for k, rec in c.items():
    if 'AR' not in k:
        continue
    r, q, m, b = parse(k)
    rows[(m, b)][(r, q)] = rec['correct']


def p2(x, y):
    if x + y == 0:
        return 1.0
    return min(binomtest(min(x, y), x + y, 0.5).pvalue * 2, 1.0)


for b in (4000, 8000, 16000):
    ub = rows[('ubcm', b)]
    ret = rows[('retrieval', b)]
    tr = rows[('uniform', b)]  # codebase uniform == recency truncation
    common = sorted(set(ub) & set(ret))
    assert len(common) == 40
    uba = sum(ub[k] for k in common) / 40
    reta = sum(ret[k] for k in common) / 40
    tra = sum(tr[k] for k in common) / 40
    bu = sum(1 for k in common if ub[k] and not ret[k])
    cu = sum(1 for k in common if ret[k] and not ub[k])
    bt = sum(1 for k in common if ub[k] and not tr[k])
    ct = sum(1 for k in common if tr[k] and not ub[k])
    print('@%d: ubcm=%.3f retrieval=%.3f trunc=%.3f | vs retrieval McNemar '
          '%d-%d p=%.4f | vs trunc %d-%d p=%.4g'
          % (b, uba, reta, tra, bu, cu, p2(bu, cu), bt, ct, p2(bt, ct)))
