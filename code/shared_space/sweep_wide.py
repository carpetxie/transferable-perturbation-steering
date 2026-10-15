"""Does the near-linear relation hold across MORE pairs and MORE tissues?

Widens the Finding 4 sweep from 3 spaces x 3 tissues to 4 spaces x 6 tissues.

Spaces
    scgpt   512-d transformer, binned expression, <cls> pooling
    gf1     Geneformer V1, 256-d, rank-value encoding, mean pooling
    gf2     Geneformer V2-104M, 768-d
    gf3     Geneformer V2-316M, 1152-d -- a fourth foundation model, added after
            the rebuild. Shares V2's vocabulary and tokenisation with gf2 but is
            3x larger, so gf2<->gf3 is a scale comparison within one lineage
    pca     128-d PCA of log-CPM over HVGs -- NOT a foundation model. This is
            the falsification control: it is by construction a purely linear
            read-out of expression, so if FM<->PCA matches FM<->FM then the
            models agree only because both linearly encode expression, and the
            'shared representation' claim is unremarkable.

Tissues
    pbmc          8 cell types (the only multi-cell-type tissue)
    thp1          AML monocyte line
    neuron        iPSC neurons, CRISPRi arm
    neuron_a      iPSC neurons, CRISPRa arm -- same cells, opposite perturbation
    k562_norman   K562 CML line
    k562_adamson  K562 again, different lab/protocol/batch

The five single-cell-type tissues are the harder test: no cluster structure, so
identity matching cannot be solved by getting the cell type right.

Two questions, and they are separate:

    Q1 (the directive)  does a linear map match or beat a 512x512 MLP, across
                        more pairs and more tissues than Finding 4 covered?
    Q2 (the control)    is FM<->FM stronger than FM<->PCA?

Every cell reports linear and MLP top-1 with spread over seeds, so a difference
inside the spread is reported as undecided rather than as a win.

The linear arm is ridge with alpha chosen by cross-validation inside the
training split (see paired_map.py). The earlier unregularised lstsq arm was not
a fair baseline: with 800 training cells and sources of 256-1152 dimensions it
was underdetermined, and it understated linear performance by up to 12x on the
highest-dimensional sources.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pca_space import get_pca          # noqa: E402
from paired_map import test_pair       # noqa: E402  ridge-CV, NOT lstsq
from sweep import embed_all            # noqa: E402

TISSUES = ["pbmc", "thp1", "neuron", "neuron_a", "k562_norman", "k562_adamson"]
SPACES = ["scgpt", "gf1", "gf2", "gf3", "pca"]
PAIRS = [(a, b) for i, a in enumerate(SPACES) for b in SPACES[i + 1:]]
N_SEED = 3


def spaces_for(tissue, n_cells=1000):
    E = embed_all(tissue, n_cells)
    out = {k: E[k].astype(np.float64) for k in ("scgpt", "gf1", "gf2", "gf3")}
    out["pca"] = get_pca(tissue, n_cells).astype(np.float64)
    n = min(len(v) for v in out.values())
    return {k: v[:n] for k, v in out.items()}


if __name__ == "__main__":
    results = {}
    for t in TISSUES:
        try:
            S = spaces_for(t)
        except Exception as e:
            print(f"=== {t}: UNAVAILABLE ({type(e).__name__}: {e})", flush=True)
            continue
        print(f"\n=== {t} ===", flush=True)
        print(f"{'pair':<16} {'linear':>16} {'mlp':>16} {'lin-mlp':>16}  verdict",
              flush=True)
        for a, b in PAIRS:
            L, M = [], []
            for s in range(N_SEED):
                r = test_pair(S[a], S[b], seed=s)
                L.append(r["linear_top1"])
                M.append(r["mlp_top1"])
            L, M = np.array(L), np.array(M)
            d = L - M
            # A difference smaller than its own spread is not a result.
            decided = abs(d.mean()) > d.std()
            verdict = ("linear" if d.mean() > 0 else "mlp") if decided else "undecided"
            results[f"{t}:{a}->{b}"] = {
                "linear": float(L.mean()), "linear_sd": float(L.std()),
                "mlp": float(M.mean()), "mlp_sd": float(M.std()),
                "diff": float(d.mean()), "diff_sd": float(d.std()),
                "verdict": verdict, "is_fm_fm": bool(a != "pca" and b != "pca")}
            print(f"{a+'->'+b:<16} {L.mean():8.3f}+/-{L.std():5.3f} "
                  f"{M.mean():8.3f}+/-{M.std():5.3f} {d.mean():+9.3f}+/-{d.std():5.3f}"
                  f"  {verdict}", flush=True)

    (HERE.parent / "artifacts" / "sweep_wide.json").write_text(
        json.dumps(results, indent=2, default=float))

    print("\n" + "=" * 66)
    dec = [v for v in results.values() if v["verdict"] != "undecided"]
    lin = [v for v in dec if v["verdict"] == "linear"]
    print(f"Q1 linear vs MLP: linear wins {len(lin)}/{len(dec)} decided cells "
          f"({len(results)-len(dec)} undecided of {len(results)})")

    # Categories are NOT pooled into one "FM<->FM" number. gf2 and gf3 are both
    # Geneformer-V2 and share a vocabulary and tokeniser, so their near-perfect
    # agreement says nothing about independent models converging. Pooling it with
    # cross-family pairs would repeat LESSONS.md rule 15.
    def cat(k):
        a, b = k.split(":")[1].split("->")
        if "pca" in (a, b):
            return "FM->PCA"
        if {a, b} == {"gf2", "gf3"}:
            return "same-lineage (GF-V2 104M vs 316M)"
        if a.startswith("gf") and b.startswith("gf"):
            return "cross-version (GF V1 vs V2)"
        return "cross-family (scGPT vs GF)"

    groups = {}
    for k, v in results.items():
        groups.setdefault(cat(k), []).append(v)
    print(f"\nQ2 by category (pooling these would be meaningless):")
    print(f"{'category':<34} {'n':>3} {'linear':>8} {'mlp':>8}")
    for c, v in sorted(groups.items()):
        L = np.mean([x["linear"] for x in v])
        M = np.mean([x["mlp"] for x in v])
        print(f"{c:<34} {len(v):3d} {L:8.3f} {M:8.3f}")
    print("\nNOTE: comparing FM->PCA to FM->FM directly is confounded by target")
    print("dimensionality (PCA is 128-d). See target_dim_control.py, which matches")
    print("the target dimension and finds FM->FM ahead on all six tissues.")
