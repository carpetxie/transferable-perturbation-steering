"""E34: does the Modal-recomputed embedding set reproduce the local results?

Everything was recomputed in one Modal container to remove cross-machine drift.
Before any new model is added on top, the recomputed set has to be checked
against what is already known, because a silent tokenisation or dtype
difference would look healthy and poison every downstream number.

TWO LEVELS OF CHECK, and the second is the one that actually matters.

  numeric      max|Z_modal - Z_local| per (condition, model). Bit-equality is
               NOT expected: local ran MPS/float32 under one torch build, Modal
               ran CUDA/float32 under another, and reduction orders differ. A
               difference around 1e-3 relative is ordinary. A difference of the
               same order as the embedding norm means something structural
               differs (wrong dictionary, wrong checkpoint, different cells).

  functional   the quantities the conclusions rest on: each model's own
               split-half reliability (its noise ceiling), and leave-one-out
               transfer. These are what must agree. Two pipelines can differ in
               the last decimal of every coordinate and still give identical
               science; they cannot differ in the ceiling and be interchangeable.

The precedent this guards against: after the /tmp wipe, a re-downloaded scGPT
checkpoint reproduced its own quality signatures (probe 0.933 vs 0.920, PR 7.1
vs 7.0) while moving scGPT->gf2 transfer from 0.532 to 0.632 across all five
seeds. Per-model health checks passed; only a cross-model quantity exposed it.
So the functional check here is deliberately a CROSS-model one.
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qlib

MODAL_EMB = pathlib.Path(os.environ.get(
    "MODAL_EMB", pathlib.Path.home() / ".cache/silico_vec2vec/modal_emb/norman"))
SHARED = ["scgpt", "gf1", "gf2", "gf3"]        # present in both pipelines
MODAL_NAME = {"scgpt": "scgpt", "gf1": "geneformer",
              "gf2": "geneformer_v2", "gf3": "geneformer_v3"}
K = 24
MIN_CELLS = 60


def load_modal(cond, model):
    f = MODAL_EMB / f"{cond.replace('/','-')}__{MODAL_NAME[model]}.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    d = np.load(f)
    return d["Z"].astype(np.float64), d["lib"], d["ngene"]


def ceilings_and_transfer(loader, idx, perts, models):
    """Per-model noise ceiling and leave-one-out transfer, in one shared space."""
    C = qlib.CONTROL
    betas, mus, ctrl = {}, {}, {}
    for m in models:
        Z, lib, ng = loader(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); ctrl[m] = Zd - mus[m]
    g = qlib.fit_gcca(ctrl, k=K)

    def S(cond, m):
        Z, lib, ng = loader(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ g["P"][m]) @ g["W"][m] @ g["C"][m]

    S0 = {m: S(C, m) for m in models}
    Sp = {p: {m: S(p, m) for m in models} for p in perts}
    out = {}
    for tgt in models:
        donors = [m for m in models if m != tgt]
        ce, tr = [], []
        for p in perts:
            Z = Sp[p][tgt]; h = len(Z) // 2
            if h < 10:
                continue
            gold = Z[h:].mean(0) - S0[tgt].mean(0)
            ce.append(qlib.cos(Z[:h].mean(0) - S0[tgt].mean(0), gold))
            d = np.mean([Sp[p][m][:len(Sp[p][m])//2].mean(0) - S0[m].mean(0)
                         for m in donors], axis=0)
            tr.append(qlib.cos(d, gold))
        out[tgt] = {"ceiling": float(np.mean(ce)), "transfer": float(np.mean(tr)),
                    "pct": float(np.mean(tr) / max(np.mean(ce), 1e-9))}
    return out


def main():
    idx, syms, _ = qlib.condition_index()
    perts = [c for c in sorted(idx.keys() - {qlib.CONTROL}) if len(idx[c]) >= MIN_CELLS]

    # ---------- level 1: numeric ----------
    print("NUMERIC: max abs difference, Modal vs local\n")
    print(f"{'model':<7} {'cond':<16} {'max|diff|':>11} {'rel to norm':>12} {'shape':>14}")
    numeric = {}
    probe = ["control"] + perts[:3]
    for m in SHARED:
        worst = 0.0
        for c in probe:
            try:
                Zm, _, _ = load_modal(c, m)
                Zl, _, _ = qlib.load_emb(c, m)
            except FileNotFoundError as e:
                print(f"{m:<7} {c:<16} MISSING {e}")
                continue
            n = min(len(Zm), len(Zl))
            d = float(np.abs(Zm[:n] - Zl[:n]).max())
            rel = d / max(float(np.linalg.norm(Zl[:n], axis=1).mean()), 1e-9)
            worst = max(worst, rel)
            print(f"{m:<7} {c:<16} {d:11.5f} {rel:12.5f} {str(Zm.shape):>14}")
        numeric[m] = worst
    print()
    for m, r in numeric.items():
        verdict = ("identical" if r < 1e-5 else
                   "float noise" if r < 5e-3 else
                   "STRUCTURAL DIFFERENCE")
        print(f"  {m:<7} worst relative diff {r:.5f}   -> {verdict}")

    # ---------- level 2: functional ----------
    print("\nFUNCTIONAL: the quantities the conclusions rest on\n")
    loc = ceilings_and_transfer(qlib.load_emb, idx, perts, SHARED)
    mod = ceilings_and_transfer(load_modal, idx, perts, SHARED)
    print(f"{'target':<7} {'ceiling loc':>12} {'ceiling mod':>12} "
          f"{'transfer loc':>13} {'transfer mod':>13} {'pct loc':>8} {'pct mod':>8}")
    ok = True
    for m in SHARED:
        dc = abs(loc[m]["ceiling"] - mod[m]["ceiling"])
        dt = abs(loc[m]["transfer"] - mod[m]["transfer"])
        ok &= (dc < 0.02 and dt < 0.02)
        print(f"{m:<7} {loc[m]['ceiling']:12.3f} {mod[m]['ceiling']:12.3f} "
              f"{loc[m]['transfer']:13.3f} {mod[m]['transfer']:13.3f} "
              f"{loc[m]['pct']*100:7.1f}% {mod[m]['pct']*100:7.1f}%")

    struct = any(r >= 5e-3 for r in numeric.values())
    print(f"\nnumeric: {'structural difference present' if struct else 'float-level only'}")
    print(f"functional: {'AGREES (all within 0.02)' if ok else 'DISAGREES'}")
    print("\nVERDICT:", "Modal set is usable; recompute everything there"
          if ok else "Modal set NOT interchangeable; do not mix pipelines")
    qlib.save_result("e34_verify_modal",
                     {"numeric_worst_rel": numeric, "local": loc, "modal": mod,
                      "functional_agrees": bool(ok)})
    print("E34_DONE")


if __name__ == "__main__":
    main()
