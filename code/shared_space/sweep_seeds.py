"""Error bars on the linear-vs-MLP comparison.

The single-seed sweep split cleanly (within-family linear 3/3, across-family
2/6) but several margins were small enough to move under resampling. This
repeats every (tissue, pair) cell over 5 train/test splits and reports the
paired difference linear - MLP with its spread, so the split can be believed
or not.
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, "src")
from sweep import embed_all, test_pair

tissues = ["pbmc", "thp1", "neuron"]
pairs = [("scgpt","gf1"), ("scgpt","gf2"), ("gf1","gf2")]
SEEDS = [0,1,2,3,4]
rows = {}
print(f"{'cell':<26} {'linear top-1':>18} {'mlp top-1':>18} {'diff (lin-mlp)':>20} {'wins':>6}")
for t in tissues:
    E = embed_all(t, 1000)
    for a,b in pairs:
        zA, zB = E[a].astype(np.float64), E[b].astype(np.float64)
        L, M = [], []
        for s in SEEDS:
            r = test_pair(zA, zB, seed=s)
            L.append(r["linear_top1"]); M.append(r["mlp_top1"])
        L, M = np.array(L), np.array(M)
        d = L - M
        key = f"{t}:{a}->{b}"
        rows[key] = {"linear_mean": float(L.mean()), "linear_sd": float(L.std()),
                     "mlp_mean": float(M.mean()), "mlp_sd": float(M.std()),
                     "diff_mean": float(d.mean()), "diff_sd": float(d.std()),
                     "wins": int((d>0).sum()), "n_seeds": len(SEEDS),
                     "family": "within" if a.startswith("gf") and b.startswith("gf") else "across"}
        print(f"{key:<26} {L.mean():8.3f} +/-{L.std():5.3f} {M.mean():8.3f} +/-{M.std():5.3f} "
              f"{d.mean():+9.3f} +/-{d.std():5.3f} {(d>0).sum():4d}/5")
json.dump(rows, open("artifacts/sweep_seeds.json","w"), indent=2, default=float)
print()
for fam in ("within","across"):
    sub=[v for v in rows.values() if v["family"]==fam]
    dm=np.array([v["diff_mean"] for v in sub]); ds=np.array([v["diff_sd"] for v in sub])
    # a cell counts as decided if its mean diff exceeds its own seed spread
    decided=[abs(m)>s for m,s in zip(dm,ds)]
    print(f"{fam:>6}-family: mean(linear-mlp) = {dm.mean():+.3f} | "
          f"cells favouring linear {sum(m>0 for m in dm)}/{len(dm)} | "
          f"cells where |diff| > seed sd: {sum(decided)}/{len(dm)}")
