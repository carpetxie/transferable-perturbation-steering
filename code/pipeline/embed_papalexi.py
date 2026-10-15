"""Embed Papalexi (THP-1 monocytes) with all four encoders.

This is the cell-type axis. Norman and Adamson are both K562; this is a
different cell type entirely, so it tests whether the multi-encoder result is a
property of the encoders or of K562.
"""
import qlib
idx, syms, _ = qlib.condition_index()
conds = sorted(idx)
print(f"dataset={qlib.DATASET} control={qlib.CONTROL} {len(conds)} conditions, "
      f"{sum(len(v) for v in idx.values())} cells", flush=True)
for enc in ["scgpt", "gf1", "gf2", "gf3"]:
    done = 0
    for c in conds:
        qlib.embed_condition(c, idx[c], syms, enc)
        done += 1
        if done % 20 == 0 or done == len(conds):
            print(f"  [{enc}] {done}/{len(conds)}", flush=True)
qlib.fit_control_pca(idx, syms)
print("PAPALEXI_EMBED_DONE", flush=True)
