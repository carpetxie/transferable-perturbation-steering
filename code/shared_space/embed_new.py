"""Embed the three added tissues, checking each loads sanely before committing."""
import sys
sys.path.insert(0,"src")
import numpy as np
from sweep import load_tissue, embed_all
from pca_space import get_pca

for t in ["k562_norman","k562_adamson","neuron_a"]:
    try:
        X,syms,lab = load_tissue(t,1000)
        X=np.asarray(X)
        det=(X>0).mean()
        print(f"{t}: {X.shape} cells x genes | mean detection {det:.3f} | "
              f"median counts {np.median(X.sum(1)):.0f} | {len(set(map(str,lab)))} label(s)",
              flush=True)
        if det < 0.001 or np.median(X.sum(1)) < 100:
            print(f"  SKIP {t}: looks degenerate, not embedding", flush=True); continue
        E=embed_all(t,1000)
        P=get_pca(t,1000)
        print(f"  embedded scgpt{E['scgpt'].shape} gf1{E['gf1'].shape} "
              f"gf2{E['gf2'].shape} pca{P.shape}", flush=True)
    except Exception as e:
        print(f"{t}: FAILED {type(e).__name__}: {e}", flush=True)
print("EMBED_NEW_DONE")
