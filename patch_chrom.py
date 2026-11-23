"""Idempotent: feature set cnzCh (gate with chromatic alpha descriptors as the topological block)."""
s = open("baselines.py", encoding="utf-8").read()
if '"cnzCh"' not in s:
    s = s.replace('         "cnz": ["comp", "mm", "logn", "cn", "znull"],',
                  '         "cnzCh": ["comp", "mm", "logn", "cn", "chrom"],\n         "cnz": ["comp", "mm", "logn", "cn", "znull"],', 1)
    s = s.replace('    for _k in ("znulld", "conn", "alttopo"):', '    for _k in ("znulld", "conn", "alttopo", "chrom"):', 1)
    assert '"cnzCh"' in s and '"alttopo", "chrom"' in s
    open("baselines.py", "w", encoding="utf-8").write(s); print("patched")
else:
    print("already")
