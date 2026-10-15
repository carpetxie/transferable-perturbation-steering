"""How much agreement survives when sequencing depth is removed from BOTH spaces?

Finding 6 showed the top shared direction is technical (|r| 0.91-0.96 with
genes-detected). So the headline r1 = 0.979-0.997 is partly the claim "both
models encode read count", which is not interesting.

The sharper measurement regresses the technical covariates out of BOTH embedding
spaces BEFORE running CCA, rather than reading them off afterwards. Removed:
log library size, log genes detected, and mitochondrial fraction, plus an
intercept. Whatever agreement remains cannot be depth.

Regression coefficients are fitted on TRAIN cells and applied to held-out cells,
so the removal itself cannot leak held-out information.

The null is unchanged: within-cell-type shuffle on PBMC, full shuffle on the
single-cell-type tissues.
"""
import sys, json
import numpy as np
sys.path.insert(0,"src")
from sweep import embed_all, load_tissue
from dimensionality import shared_spectrum

def deconfound(Z, C, tr):
    """Remove the span of C (fitted on tr) from Z."""
    Ct=np.hstack([C, np.ones((len(C),1))])
    beta=np.linalg.lstsq(Ct[tr], Z[tr], rcond=None)[0]
    return Z - Ct@beta

def within_type_shuffle(y, rng):
    idx=np.arange(len(y))
    for t in np.unique(y):
        m=idx[y==t]; idx[y==t]=rng.permutation(m)
    return idx

res={}
print(f"{'cell':<22} {'r1 raw':>8} {'r1 depth-free':>14} {'null max':>9} {'dims>null':>9}")
for t in ["pbmc","thp1","neuron"]:
    X,syms,lab=load_tissue(t,1000); X=np.asarray(X,dtype=np.float64)
    lib=X.sum(1); ng=(X>0).sum(1)
    mito=[i for i,s in enumerate(syms) if str(s).upper().startswith(("MT-","MT."))]
    mf=X[:,mito].sum(1)/np.maximum(lib,1) if mito else np.zeros(len(X))
    C=np.column_stack([np.log1p(lib), np.log1p(ng), mf])
    E=embed_all(t,1000)
    y=np.array([{v:i for i,v in enumerate(np.unique(lab))}[c] for c in lab])
    multi=len(np.unique(y))>1
    for a,b in [("scgpt","gf2"),("gf1","gf2")]:
        A=E[a].astype(np.float64); B=E[b].astype(np.float64)
        raw=np.mean([shared_spectrum(A,B,seed=s)["r1"] for s in range(3)])
        reals=[]; nulls=[]
        for s in range(3):
            rng=np.random.default_rng(s); tr=rng.permutation(len(A))[:int(0.8*len(A))]
            Ad,Bd=deconfound(A,C,tr),deconfound(B,C,tr)
            reals.append(shared_spectrum(Ad,Bd,seed=s)["r"])
        real=np.mean(reals,axis=0)
        for s in range(8):
            rng=np.random.default_rng(4000+s); tr=rng.permutation(len(A))[:int(0.8*len(A))]
            Ad,Bd=deconfound(A,C,tr),deconfound(B,C,tr)
            perm=within_type_shuffle(y,rng) if multi else rng.permutation(len(B))
            nulls.append(shared_spectrum(Ad,Bd[perm],seed=s%3)["r"])
        L=min(len(real),min(len(x) for x in nulls))
        thr=np.array([x[:L] for x in nulls]).max(0)
        ab=real[:L]>thr; k=int(np.argmin(ab)) if not ab.all() else L
        key=f"{t}:{a}->{b}"
        res[key]={"r1_raw":float(raw),"r1_depthfree":float(real[0]),
                  "null_max":float(thr[0]),"dims_above_null":k,
                  "r5":float(real[4]),"r10":float(real[9])}
        print(f"{key:<22} {raw:8.3f} {real[0]:14.3f} {thr[0]:9.3f} {k:9d}")
json.dump(res,open("artifacts/depth_free.json","w"),indent=2,default=float)
# NOTE: an earlier version of this summary pooled min(real) against max(null)
# across ALL cells. That violates the LESSONS.md rule against averaging
# non-exchangeable nulls: PBMC's null is the WITHIN-CELL-TYPE shuffle, which
# keeps cell type and is therefore high (~0.97) by construction, while the
# single-cell-type tissues use a full shuffle whose null is ~0.1. Pooling them
# made a clean result read as "largely technical". Each tissue is judged against
# its own null.
for k, v in res.items():
    gap = v["r1_depthfree"] - v["null_max"]
    print(f"  {k:<22} depth-free r1 {v['r1_depthfree']:.3f} vs own null "
          f"{v['null_max']:.3f}  gap {gap:+.3f}")
single = {k: v for k, v in res.items() if not k.startswith("pbmc")}
r = [v["r1_depthfree"] for v in single.values()]
n = [v["null_max"] for v in single.values()]
print(f"\nsingle-cell-type tissues (full-shuffle null, no cluster shortcut):")
print(f"  depth-free r1 {min(r):.3f}-{max(r):.3f}   worst null {max(n):.3f}")
print("VERDICT:", "agreement survives depth removal" if min(r) > max(n) + 0.3
      else "agreement was largely technical")
print("PBMC is reported separately: its within-type null is ~0.97 by design, so")
print("its top direction remains cell-type driven even after depth removal.")
