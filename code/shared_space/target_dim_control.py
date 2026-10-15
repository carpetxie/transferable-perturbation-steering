"""Is FM->PCA competitive only because PCA is a 128-d target?

The wide sweep shows scGPT->PCA matching or beating scGPT->Geneformer on 3 of 6
tissues. Before concluding that FM-FM agreement is no better than agreement with
a linear read-out of expression, the obvious confound must be removed: PCA is
128-dimensional and Geneformer-V2 is 768-dimensional, and a lower-dimensional
target may simply be easier to predict.

Control: compress the Geneformer target to exactly 128 dimensions with PCA, so
the two targets have identical dimensionality, and re-run the same ridge-CV map.

    if scgpt->gf2_128 jumps to ~scgpt->pca
        target dimensionality explains the result, and the PCA comparison says
        nothing about shared representation
    if scgpt->gf2_128 stays near scgpt->gf2
        dimensionality is not the driver and the comparison stands
"""
import sys, json
import numpy as np
sys.path.insert(0,"src")
from sweep import embed_all
from pca_space import get_pca
from paired_map import test_pair

def to_dim(Z,k=128):
    Z=Z-Z.mean(0)
    U,S,_=np.linalg.svd(Z,full_matrices=False)
    return U[:,:k]*S[:k]

TIS=["pbmc","thp1","neuron","neuron_a","k562_norman","k562_adamson"]
print(f"{'tissue':<14} {'scgpt->gf2':>11} {'scgpt->gf2_128':>15} {'scgpt->pca':>11}")
rows={}
for t in TIS:
    E=embed_all(t,1000); A=E["scgpt"].astype(np.float64)
    B=E["gf2"].astype(np.float64); P=get_pca(t,1000).astype(np.float64)
    B128=to_dim(B,128)
    f=lambda X,Y: float(np.mean([test_pair(X,Y,seed=s,with_mlp=False)["linear_top1"]
                                 for s in range(3)]))
    r=(f(A,B), f(A,B128), f(A,P))
    rows[t]=r
    print(f"{t:<14} {r[0]:11.3f} {r[1]:15.3f} {r[2]:11.3f}")
json.dump(rows,open("artifacts/target_dim_control.json","w"),indent=2)
a=np.array([r[0] for r in rows.values()]); b=np.array([r[1] for r in rows.values()])
c=np.array([r[2] for r in rows.values()])
print(f"\nmean  gf2 {a.mean():.3f}   gf2_128 {b.mean():.3f}   pca {c.mean():.3f}")
print(f"compressing the target to 128d changes FM->FM by {b.mean()-a.mean():+.3f}")
print("VERDICT:", "dimensionality DOES explain the PCA result"
      if abs(b.mean()-c.mean())<0.05 and b.mean()-a.mean()>0.05
      else "dimensionality does NOT explain it")
