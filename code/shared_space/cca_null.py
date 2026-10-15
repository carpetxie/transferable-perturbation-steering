"""Is the held-out canonical correlation of ~0.99 real, or an overfitting artifact?

CCA between a 512-d and a 768-d space fitted on 800 cells is exactly the regime
where canonical correlations are inflated to 1.0 by construction: with enough
free directions, ANY two point clouds can be rotated into agreement in-sample.
Held-out evaluation is supposed to defend against this, but held-out sets of 200
cells can still show high correlation if the canonical directions latch onto a
few high-leverage cells.

So the number is calibrated against a null where the answer is KNOWN to be zero:
the same cells, the same dimensions, the same fitting procedure, but with the
row correspondence between A and B destroyed by a shuffle. Any correlation the
null produces is manufactured by the estimator rather than present in the data.

This follows the comparator rules in Claude/LESSONS.md: the null runs on the same
population and the same pipeline, and it is calibrated over a DISTRIBUTION of
draws rather than a single one.

A second, harder control keeps the pairing but shuffles WITHIN cell type, so
correspondence at the cell-type level survives while per-cell identity is
destroyed. That separates 'the spaces agree about cell types' from 'the spaces
agree about individual cells'.
"""
import sys, json
import numpy as np
sys.path.insert(0,"src")
from sweep import embed_all
from dimensionality import shared_spectrum

def within_type_shuffle(y, rng):
    idx=np.arange(len(y))
    for t in np.unique(y):
        m=idx[y==t]; idx[y==t]=rng.permutation(m)
    return idx

res={}
print(f"{'cell':<22} {'REAL r1':>16} {'shuffled r1':>16} {'within-type r1':>16}")
for t in ["pbmc","thp1","neuron"]:
    E=embed_all(t,1000); lab=E["labels"]
    y=np.array([{v:i for i,v in enumerate(np.unique(lab))}[c] for c in lab])
    for a,b in [("scgpt","gf1"),("scgpt","gf2"),("gf1","gf2")]:
        A=E[a].astype(np.float64); B=E[b].astype(np.float64)
        real=[shared_spectrum(A,B,seed=s)["r1"] for s in range(3)]
        null=[]; wt=[]
        for s in range(8):
            rng=np.random.default_rng(1000+s)
            null.append(shared_spectrum(A, B[rng.permutation(len(B))], seed=s%3)["r1"])
            if len(np.unique(y))>1:
                wt.append(shared_spectrum(A, B[within_type_shuffle(y,rng)], seed=s%3)["r1"])
        key=f"{t}:{a}->{b}"
        res[key]={"real":float(np.mean(real)),"real_sd":float(np.std(real)),
                  "shuffled":float(np.mean(null)),"shuffled_sd":float(np.std(null)),
                  "shuffled_max":float(np.max(null)),
                  "within_type":float(np.mean(wt)) if wt else None}
        w=f"{np.mean(wt):8.3f}+/-{np.std(wt):5.3f}" if wt else "        n/a     "
        print(f"{key:<22} {np.mean(real):8.3f}+/-{np.std(real):5.3f} "
              f"{np.mean(null):8.3f}+/-{np.std(null):5.3f} {w}")
json.dump(res, open("artifacts/cca_null.json","w"), indent=2, default=float)
mx=max(v["shuffled_max"] for v in res.values()); mn=min(v["real"] for v in res.values())
print(f"\nworst REAL {mn:.3f}   |   best SHUFFLED {mx:.3f}")
print("VERDICT:", "REAL - null cannot reach it" if mn>mx else
      "ARTIFACT - null reaches the real value, estimator manufactures it")
