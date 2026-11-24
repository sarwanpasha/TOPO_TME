"""Idempotent: fsets cnz32 (TopoNull with 32 permutations) and cnzPt (TopoNull on perturbed maps)."""
s = open("baselines.py", encoding="utf-8").read()
if '"cnz32"' not in s:
    s = s.replace('         "cnz": ["comp", "mm", "logn", "cn", "znull"],',
                  '         "cnz32": ["comp", "mm", "logn", "cn", "znull32"], "cnzPt": ["comp", "mm", "logn", "cn", "znullp"],\n'
                  '         "cnz": ["comp", "mm", "logn", "cn", "znull"],', 1)
    s = s.replace('"alttopo", "chrom", "moran"):', '"alttopo", "chrom", "moran", "znull32", "znullp"):', 1)
    assert '"cnz32"' in s and '"znull32", "znullp"' in s
    open("baselines.py", "w", encoding="utf-8").write(s); print("patched")
else:
    print("already")
