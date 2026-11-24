"""Idempotent: fsets cnzNC (topology block = TopoNull marker fields of negative-control channels only: Histone in
Basel/Zurich, DNA1 in luad_r0)."""
s = open("baselines.py", encoding="utf-8").read()
if '"cnzNC"' not in s:
    s = s.replace('         "cnz": ["comp", "mm", "logn", "cn", "znull"],',
                  '         "cnzNC": ["comp", "mm", "logn", "cn", "znullnc"],\n         "cnz": ["comp", "mm", "logn", "cn", "znull"],', 1)
    s = s.replace('"znull32", "znullp"):', '"znull32", "znullp", "znullnc"):', 1)
    assert '"cnzNC"' in s and '"znullnc"):' in s
    open("baselines.py", "w", encoding="utf-8").write(s); print("patched")
else:
    print("already")
