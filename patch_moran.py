"""Idempotent: feature sets with the spatial-autocorrelation baseline.
  cnzMI   calibrated Moran/Geary block     cnzMIr  raw Moran/Geary block     cnzMIT  calibrated Moran/Geary + TopoNull"""
s = open("baselines.py", encoding="utf-8").read()
if '"cnzMI"' not in s:
    s = s.replace('         "cnz": ["comp", "mm", "logn", "cn", "znull"],',
                  '         "cnzMI": ["comp", "mm", "logn", "cn", "moranz"], "cnzMIr": ["comp", "mm", "logn", "cn", "moranr"],\n'
                  '         "cnzMIT": ["comp", "mm", "logn", "cn", "moranz", "znull"],\n'
                  '         "cnz": ["comp", "mm", "logn", "cn", "znull"],', 1)
    s = s.replace('    for _k in ("znulld", "conn", "alttopo", "chrom"):', '    for _k in ("znulld", "conn", "alttopo", "chrom", "moran"):', 1)
    s = s.replace('    if "conn" in fe:\n',
                  '    if "moran" in fe:\n'
                  '        _h = fe["moran"].shape[1] // 2; fe["moranr"], fe["moranz"] = fe["moran"][:, :_h], fe["moran"][:, _h:]\n'
                  '    if "conn" in fe:\n', 1)
    assert '"cnzMI"' in s and 'fe["moranz"]' in s and '"chrom", "moran"' in s
    open("baselines.py", "w", encoding="utf-8").write(s); print("patched")
else:
    print("already")
