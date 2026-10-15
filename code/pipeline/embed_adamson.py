"""Embed the Adamson conditions for external validation of the Norman findings."""
import sys
import numpy as np
import qlib

idx, syms, triples = qlib.condition_index()
conds = sorted(idx)
print(f"dataset={qlib.DATASET} control={qlib.CONTROL} "
      f"{len(conds)} conditions, {sum(len(v) for v in idx.values())} cells",
      flush=True)
for enc in qlib.ENCODERS:
    done = 0
    for c in conds:
        qlib.embed_condition(c, idx[c], syms, enc)
        done += 1
        if done % 20 == 0 or done == len(conds):
            print(f"  [{enc}] {done}/{len(conds)}", flush=True)
qlib.fit_control_pca(idx, syms)
for c in conds:
    qlib.pca_embed_condition(c, idx, syms)
print("ADAMSON_EMBED_DONE", flush=True)
