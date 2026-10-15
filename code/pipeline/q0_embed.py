"""Stage 0: embed every Norman condition with every encoder, checkpointed.

Restartable by construction: each (condition, encoder) lands in its own .npz the
moment it finishes, so an interrupt loses at most one condition. Progress prints
one line per condition so the queue log shows exactly where a crash happened.

Also builds the control-fitted PCA basis and the per-condition gene-space means
(for q5), and finishes with an integrity check that every expected file exists
and none is empty or NaN.
"""
import sys
import numpy as np
import qlib

def main():
    idx, syms, triples = qlib.condition_index()
    conds = sorted(idx.keys())
    n_cells = sum(len(v) for v in idx.values())
    print(f"{len(conds)} conditions, {len(triples)} triples, {n_cells} cells, "
          f"encoders {qlib.ENCODERS}", flush=True)

    # ---- FM embeddings, per condition, per encoder (checkpointed inside)
    for enc in qlib.ENCODERS:
        done = 0
        for c in conds:
            qlib.embed_condition(c, idx[c], syms, enc)
            done += 1
            if done % 20 == 0 or done == len(conds):
                print(f"  [{enc}] {done}/{len(conds)}", flush=True)

    # ---- PCA space and gene-space means
    qlib.fit_control_pca(idx, syms)
    for i, c in enumerate(conds):
        qlib.pca_embed_condition(c, idx, syms)
        qlib.gene_space_mean(c, idx, syms)
        if (i + 1) % 40 == 0:
            print(f"  [pca+gene] {i+1}/{len(conds)}", flush=True)

    # ---- integrity check
    bad = []
    for c in conds:
        for enc in qlib.ENCODERS + ["pca"]:
            try:
                Z, lib, ng = qlib.load_emb(c, enc) if enc != "pca" else \
                    qlib.pca_embed_condition(c, idx, syms)
                if len(Z) == 0 or np.isnan(Z).any():
                    bad.append((c, enc, "empty-or-nan"))
            except Exception as e:
                bad.append((c, enc, type(e).__name__))
    if bad:
        print("INTEGRITY FAILURES:", bad[:10], flush=True)
        sys.exit(1)
    qlib.save_result("q0_manifest", {
        "n_conditions": len(conds), "n_triples": len(triples),
        "n_cells": n_cells, "encoders": qlib.ENCODERS + ["pca"],
        "triples": [list(t) for t in triples]})
    print("Q0_DONE", flush=True)

if __name__ == "__main__":
    main()
