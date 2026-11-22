"""Summarise TopoGate's choices (results/gatelog/*.jsonl): how often topology is switched on (alpha>0), the chosen
base learner, mean alpha, and the inner-CV AUROC gain of the chosen (base, alpha) over the best alpha=0 option."""
import os, glob, json
import numpy as np, pandas as pd
import common as C
R = [json.loads(l) for f in glob.glob(os.path.join(C.RES, "gatelog", "*.jsonl")) for l in open(f)]
df = pd.DataFrame(R)
df["cohort"] = np.where(df["external"], df["cohort"] + "_ext", df["cohort"])
df["topo_on"] = df["alpha"] > 0
df["cn_base"] = df["base"].str.startswith("cmcn")
df["inner_gain"] = [r["scores"][f"{r['base']}|{r['alpha']}"] - max(v for k, v in r["scores"].items() if k.endswith("|0.0"))
                    for r in R]
df = df.drop_duplicates(["cohort", "task", "fold", "seed"])
g = df.groupby(["cohort", "task"]).agg(n=("alpha", "size"), topo_on=("topo_on", "mean"), mean_alpha=("alpha", "mean"),
                                       cn_base=("cn_base", "mean"), inner_gain=("inner_gain", "mean")).reset_index()
pd.set_option("display.width", 200)
print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nOVERALL runs", len(df), "topology on", f"{df.topo_on.mean():.3f}", "CN base", f"{df.cn_base.mean():.3f}",
      "alpha dist", df["alpha"].value_counts().sort_index().to_dict(), "base dist", df["base"].value_counts().to_dict())
pt = os.path.join(C.ROOT, "paired_tests.csv")
if os.path.exists(pt):
    P = pd.read_csv(pt); P = P[(P.ours == "lgbmgate4_cnz") & (P.prior == "lgbmcv_comp_mm")][["cohort", "task", "diff"]]
    m = g.merge(P, on=["cohort", "task"], how="left")
    print("\nSpearman(topo_on, test gain vs tuned comp):", m[["topo_on", "diff"]].corr("spearman").iloc[0, 1])
    print(m[["cohort", "task", "topo_on", "diff"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
g.to_csv(os.path.join(C.RES, "interp", "gate_summary.csv"), index=False)
