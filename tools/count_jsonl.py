#!/usr/bin/env python3
import sys, os, glob, json

base = os.path.join('training', 'data')
files = []
if os.path.isdir(base):
    files = sorted(glob.glob(os.path.join(base, '*.jsonl')))

total = 0
rows = []
for p in files:
    try:
        n = sum(1 for _ in open(p, 'r', encoding='utf-8', errors='ignore'))
    except Exception:
        n = 0
    total += n
    rows.append((os.path.basename(p), n))

print(json.dumps({
    'files': [{'file': f, 'samples': n} for f, n in rows],
    'total_samples': total
}, ensure_ascii=False))


