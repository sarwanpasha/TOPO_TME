"""Write a feature file with the TopoNull marker-field columns of selected markers only (negative-control channels).
python make_subset.py <cohort> <name> <marker> [<marker> ...]  -> data/<cohort>/feats_<name>.npz {<name>, image_id}"""
import os, sys
import numpy as np
import common as C
import feat_null as FN

c, name, mk = sys.argv[1], sys.argv[2], sys.argv[3:]
_, _, meta = C.load_cohort(c)
nm = np.array(FN.feature_names(meta["markers"], meta["types"]))
z = np.load(os.path.join(C.cohort_dir(c), "feats_znull.npz"))
X = z["znull"]
assert X.shape[1] == len(nm), (X.shape, len(nm))
sel = np.array([any(n.startswith(f"zM:{m}|") for m in mk) for n in nm])
np.savez_compressed(os.path.join(C.cohort_dir(c), f"feats_{name}.npz"), **{name: X[:, sel]}, image_id=z["image_id"])
print(c, name, "columns", int(sel.sum()), "of", len(nm))
