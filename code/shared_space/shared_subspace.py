"""How many dimensions of PER-CELL agreement exist between two foundation models?

cca_null.py established that the top canonical correlation (~0.99) is real: a
row-shuffled null never exceeds 0.161. But it also showed that on PBMC the top
direction survives a WITHIN-CELL-TYPE shuffle at 0.97, so there it is largely
cell-type agreement rather than per-cell agreement.

This counts, dimension by dimension, how many canonical directions carry
agreement that the appropriate null cannot produce:

    single-cell-type tissues (THP-1, neuron)
        the only available null is the full row shuffle, and because every cell
        has the same label, that null already destroys nothing BUT per-cell
        identity. Dimensions above it are per-cell.

    PBMC
        the full shuffle is too weak a null -- it also destroys cell type. The
        honest null is the WITHIN-CELL-TYPE shuffle, which keeps cell type and
        removes only per-cell identity. Dimensions above THAT are per-cell.

The threshold is the null's maximum over draws, not its mean, so a dimension
counts only if no null draw reached it.
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

N_NULL=12
res={}
print(f"{'cell':<22} {'null type':<12} {'dims>null':>9} {'r at that dim':>14} {'null max':>9}")
for t in ["pbmc","thp1","neuron"]:
    E=embed_all(t,1000); lab=E["labels"]
    y=np.array([{v:i for i,v in enumerate(np.unique(lab))}[c] for c in lab])
    multi = len(np.unique(y))>1
    for a,b in [("scgpt","gf1"),("scgpt","gf2"),("gf1","gf2")]:
        A=E[a].astype(np.float64); B=E[b].astype(np.float64)
        # Spectra can differ in length across seeds: the PCA basis is refitted
        # per split and _reduce keeps however many directions carry 99% of the
        # variance, capped at min(128, n/4). Averaging them directly raised
        # "setting an array element with a sequence ... inhomogeneous shape", so
        # every spectrum is truncated to the shortest before comparison. This is
        # the conservative choice: it can only shorten the reported leading run.
        reals=[shared_spectrum(A,B,seed=s)["r"] for s in range(3)]
        nulls=[]
        for s in range(N_NULL):
            rng=np.random.default_rng(2000+s)
            perm = within_type_shuffle(y,rng) if multi else rng.permutation(len(B))
            nulls.append(shared_spectrum(A,B[perm],seed=s%3)["r"])
        L=min(min(len(x) for x in reals), min(len(x) for x in nulls))
        real=np.mean([x[:L] for x in reals],axis=0)
        thr=np.array([x[:L] for x in nulls]).max(0)      # per-dimension null max
        above=real>thr
        k=int(np.argmin(above)) if not above.all() else len(above)  # leading run
        key=f"{t}:{a}->{b}"
        res[key]={"null":"within-type" if multi else "full-shuffle",
                  "dims_above_null":k,"r_at_k":float(real[k-1]) if k else None,
                  "null_max_at_k":float(thr[k-1]) if k else None,
                  "real":real.tolist(),"null_max":thr.tolist()}
        print(f"{key:<22} {res[key]['null']:<12} {k:9d} "
              f"{(real[k-1] if k else float('nan')):14.3f} {(thr[k-1] if k else float('nan')):9.3f}")
json.dump(res, open("artifacts/shared_subspace.json","w"), indent=2, default=float)
print("\nper-cell shared dimensions (leading run above a null that keeps cell type):")
for k,v in res.items(): print(f"  {k:<22} {v['dims_above_null']:3d}")
