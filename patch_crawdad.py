"""Idempotent: feature sets with the CRAWDAD-style multiscale co-localization baseline.
  cnzCW   calibrated multiscale co-localization block     cnzCWT  co-localization + TopoNull"""
s = open("baselines.py", encoding="utf-8").read()
if '"cnzCW"' not in s:
    s = s.replace('         "cnz": ["comp", "mm", "logn", "cn", "znull"],',
                  '         "cnzCW": ["comp", "mm", "logn", "cn", "crawdad"], "cnzCWT": ["comp", "mm", "logn", "cn", "crawdad", "znull"],\n'
                  '         "cnz": ["comp", "mm", "logn", "cn", "znull"],', 1)
    s = s.replace('"moran", "znull32"', '"moran", "crawdad", "znull32"', 1)
    assert '"cnzCW"' in s and '"moran", "crawdad"' in s
    open("baselines.py", "w", encoding="utf-8").write(s); print("patched")
else:
    print("already")
