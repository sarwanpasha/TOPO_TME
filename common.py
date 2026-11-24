"""Shared utilities: paths, cache I/O, folds, metrics, Cox loss."""
import os, json, pickle
import numpy as np
import pandas as pd

ROOT = os.environ.get("TOPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# task registry: name -> (kind, column(s))
#   kind 'bin': column with 0/1 ; kind 'surv': (time_col, event_col)
TASKS = {
    "danenberg": {"ER": ("bin", "ER"), "HER2": ("bin", "HER2"), "PR": ("bin", "PR"),
                  "basal": ("bin", "basal"), "grade3": ("bin", "grade3"),
                  "OS": ("surv", ("os_time", "os_event")), "DSS": ("surv", ("os_time", "dss_event")),
                  "RFS": ("surv", ("rfs_time", "rfs_event"))},
    "basel": {"ER": ("bin", "ER"), "HER2": ("bin", "HER2"), "PR": ("bin", "PR"),
              "TNBC": ("bin", "TNBC"), "grade3": ("bin", "grade3"),
              "OS": ("surv", ("os_time", "os_event")), "DSS": ("surv", ("os_time", "dss_event")),
              "DFS": ("surv", ("dfs_time", "dfs_event"))},
    "zurich": {"ER": ("bin", "ER"), "HER2": ("bin", "HER2"), "PR": ("bin", "PR"),
               "TNBC": ("bin", "TNBC"), "grade3": ("bin", "grade3")},
    "codex": {"DII": ("bin", "DII"), "OS": ("surv", ("os_time", "os_event"))},
    "luad": {"progression": ("bin", "progression"), "stage34": ("bin", "stage34"), "solid": ("bin", "solid"),
             "OS": ("surv", ("os_time", "os_event"))},
    "hoch": {"response": ("bin", "response"), "relapse": ("bin", "relapse"), "death": ("bin", "death")},
    "luad_r0": {"progression": ("bin", "progression"), "stage34": ("bin", "stage34"), "solid": ("bin", "solid")},
    "dryad": {"OS": ("surv", ("os_time", "os_event")), "lusc": ("bin", "lusc"), "io_response": ("bin", "io_response")},
    "cords": {"OS": ("surv", ("os_time", "os_event")), "DFS": ("surv", ("dfs_time", "dfs_event")), "relapse": ("bin", "relapse"), "lnmet": ("bin", "lnmet"), "grade3": ("bin", "grade3"), "squamous": ("bin", "squamous")},
    "synth_ring": {"y": ("bin", "y")},
    "synth_interact": {"y": ("bin", "y")},
    "synth_channel": {"y": ("bin", "y")},
    "synth_ringmk": {"y": ("bin", "y")},
}
EXTERNAL = {"basel": "zurich"}  # train on key, test on value (shared panel/vocab)


def cohort_dir(c):
    return os.path.join(DATA, c)


def load_cohort(c):
    d = cohort_dir(c)
    with open(os.path.join(d, "images.pkl"), "rb") as f:
        images = pickle.load(f)
    pat = pd.read_csv(os.path.join(d, "patients.csv"), dtype={"patient_id": str})
    meta = json.load(open(os.path.join(d, "meta.json")))
    for im in images:
        im["patient_id"] = str(im["patient_id"])
    return images, pat, meta


def task_frame(pat, cohort, task):
    kind, col = TASKS[cohort][task]
    if kind == "bin":
        df = pat[["patient_id", col]].dropna().copy()
        df["y"] = df[col].astype(int)
        if df["y"].nunique() < 2:
            return None, kind
        return df[["patient_id", "y"]].reset_index(drop=True), kind
    t, e = col
    if t not in pat or e not in pat:
        return None, kind
    df = pat[["patient_id", t, e]].dropna().copy()
    df = df[df[t] > 0]
    df.columns = ["patient_id", "time", "event"]
    df["event"] = df["event"].astype(int)
    if df["event"].sum() < 10:
        return None, kind
    return df.reset_index(drop=True), kind


def make_folds(df, kind, n_folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    strat = df["y"].values if kind == "bin" else df["event"].values
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(df)), strat)]


def inner_split(df, tr_idx, kind, seed, frac=0.2):
    from sklearn.model_selection import StratifiedShuffleSplit
    strat = (df["y"].values if kind == "bin" else df["event"].values)[tr_idx]
    sss = StratifiedShuffleSplit(1, test_size=frac, random_state=seed)
    a, b = next(sss.split(np.zeros(len(tr_idx)), strat))
    return tr_idx[a], tr_idx[b]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def cindex(time, event, risk):
    """Harrell's C (higher risk -> earlier event)."""
    time = np.asarray(time, float); event = np.asarray(event, int); risk = np.asarray(risk, float)
    order = np.argsort(time)
    t, e, r = time[order], event[order], risk[order]
    num = 0.0; den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        later = t > t[i]
        den += later.sum()
        num += (r[i] > r[later]).sum() + 0.5 * (r[i] == r[later]).sum()
    return float(num / den) if den > 0 else float("nan")


def metric(kind, df, pred):
    if kind == "bin":
        return auroc(df["y"].values, pred)
    return cindex(df["time"].values, df["event"].values, pred)


def cox_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow), torch tensors."""
    import torch
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    lse = torch.logcumsumexp(r, 0)
    n_ev = e.sum().clamp(min=1.0)
    return -((r - lse) * e).sum() / n_ev


def result_path(cohort, task, method, fold, seed, tag=""):
    d = os.path.join(RES, cohort + tag, task, method)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"f{fold}_s{seed}.json")


