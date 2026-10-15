"""E39: add control-trained scVI and PCA to the fixed shared space.

This is deliberately a new result rather than a rewrite of E33. E34 first
decides whether the Modal recomputation preserves the established four-model
result. Only then does this test fit one six-view GCCA space on control cells
and ask whether perturbation effects transfer into and out of two
architecturally independent representations.

scVI and PCA are trained/fitted on control cells only. Target perturbation
labels are used only to construct the held-out evaluation half and its
split-half noise ceiling. Cell halves are identical across every encoder so a
donor can never observe the target's gold physical cells through another view.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qlib

MODAL_EMB = pathlib.Path(os.environ.get(
    "MODAL_EMB", pathlib.Path.home() / ".cache/silico_vec2vec/modal_emb/norman"))
MODELS = ["scgpt", "gf1", "gf2", "gf3", "scvi", "pca"]
ORIGINAL = {"scgpt", "gf1", "gf2", "gf3"}
MODAL_NAME = {
    "scgpt": "scgpt",
    "gf1": "geneformer",
    "gf2": "geneformer_v2",
    "gf3": "geneformer_v3",
    "scvi": "scvi",
    "pca": "pca",
}
K = 24
MIN_CELLS = 60


def load_modal(cond, model):
    path = MODAL_EMB / f"{cond.replace('/', '-')}__{MODAL_NAME[model]}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path)
    return d["Z"].astype(np.float64), d["lib"], d["ngene"]


def random_maps(ctrl, k, seed=0):
    """Independent matched-dimensional bases: projection without alignment."""
    rng = np.random.default_rng(seed)
    maps = {}
    for model, Z in ctrl.items():
        q, _ = np.linalg.qr(rng.normal(size=(Z.shape[1], min(k, Z.shape[1]))))
        maps[model] = q
    return maps


def main():
    idx, _, _ = qlib.condition_index()
    control = qlib.CONTROL
    perts = [c for c in sorted(idx.keys() - {control})
             if len(idx[c]) >= MIN_CELLS]

    betas, mus, ctrl = {}, {}, {}
    for model in MODELS:
        Z, lib, ngene = load_modal(control, model)
        beta, _ = qlib.fit_depth_projector(Z, lib, ngene)
        Zd = qlib.apply_depth_projector(Z, lib, ngene, beta)
        betas[model] = beta
        mus[model] = Zd.mean(0)
        ctrl[model] = Zd - mus[model]

    shared = qlib.fit_gcca(ctrl, k=K)
    random = random_maps(ctrl, K)

    def projected(cond, model):
        Z, lib, ngene = load_modal(cond, model)
        Zd = qlib.apply_depth_projector(Z, lib, ngene, betas[model]) - mus[model]
        return (Zd @ shared["P"][model]) @ shared["W"][model] @ shared["C"][model]

    def random_projected(cond, model):
        Z, lib, ngene = load_modal(cond, model)
        Zd = qlib.apply_depth_projector(Z, lib, ngene, betas[model]) - mus[model]
        return Zd @ random[model]

    S0 = {m: projected(control, m) for m in MODELS}
    R0 = {m: random_projected(control, m) for m in MODELS}
    Sp = {p: {m: projected(p, m) for m in MODELS} for p in perts}
    Rp = {p: {m: random_projected(p, m) for m in MODELS} for p in perts}

    rows = []
    for target in MODELS:
        donor_sets = {
            "all": [m for m in MODELS if m != target],
            "original": [m for m in MODELS if m != target and m in ORIGINAL],
            "new": [m for m in MODELS if m != target and m not in ORIGINAL],
        }
        scores = {name: [] for name, donors in donor_sets.items() if donors}
        mismatch = {name: [] for name, donors in donor_sets.items() if donors}
        random_scores = {name: [] for name, donors in donor_sets.items() if donors}
        native, ceiling = [], []
        for j, pert in enumerate(perts):
            target_cells = Sp[pert][target]
            half = len(target_cells) // 2
            if half < 10:
                continue
            gold = target_cells[half:].mean(0) - S0[target].mean(0)
            native.append(qlib.cos(target_cells[:half].mean(0) - S0[target].mean(0),
                                   gold))
            ceiling.append(native[-1])
            wrong = perts[(j + 1) % len(perts)]
            random_gold = (Rp[pert][target][half:].mean(0)
                           - R0[target].mean(0))
            for name, donors in donor_sets.items():
                if not donors:
                    continue
                delta = np.mean([
                    Sp[pert][m][:len(Sp[pert][m]) // 2].mean(0) - S0[m].mean(0)
                    for m in donors], axis=0)
                wrong_delta = np.mean([
                    Sp[wrong][m][:len(Sp[wrong][m]) // 2].mean(0) - S0[m].mean(0)
                    for m in donors], axis=0)
                random_delta = np.mean([
                    Rp[pert][m][:len(Rp[pert][m]) // 2].mean(0) - R0[m].mean(0)
                    for m in donors], axis=0)
                scores[name].append(qlib.cos(delta, gold))
                mismatch[name].append(qlib.cos(wrong_delta, gold))
                random_scores[name].append(qlib.cos(random_delta, random_gold))

        ce = float(np.mean(ceiling))
        row = {
            "target": target,
            "ceiling": ce,
            "native": float(np.mean(native)),
            "arms": {},
        }
        for name in scores:
            transfer = float(np.mean(scores[name]))
            row["arms"][name] = {
                "donors": donor_sets[name],
                "transfer": transfer,
                "pct_ceiling": transfer / max(ce, 1e-9),
                "mismatch_floor": float(np.mean(mismatch[name])),
                "random_basis": float(np.mean(random_scores[name])),
            }
        rows.append(row)

    print(f"six-view fixed GCCA, K={K}, perturbations={len(perts)}")
    print(f"{'target':<7} {'arm':<9} {'transfer':>9} {'ceiling':>9} "
          f"{'%ceil':>8} {'mismatch':>9} {'random':>9}")
    for row in rows:
        for name, arm in row["arms"].items():
            print(f"{row['target']:<7} {name:<9} {arm['transfer']:9.3f} "
                  f"{row['ceiling']:9.3f} {100 * arm['pct_ceiling']:7.1f}% "
                  f"{arm['mismatch_floor']:9.3f} {arm['random_basis']:9.3f}")

    new_targets = [r for r in rows if r["target"] in {"scvi", "pca"}]
    old_targets = [r for r in rows if r["target"] in ORIGINAL]
    summary = {
        "new_target_original_donors_pct": float(np.mean([
            r["arms"]["original"]["pct_ceiling"] for r in new_targets])),
        "old_target_new_donors_pct": float(np.mean([
            r["arms"]["new"]["pct_ceiling"] for r in old_targets])),
        "all_donors_pct": float(np.mean([
            r["arms"]["all"]["pct_ceiling"] for r in rows])),
    }
    out = {
        "models": MODELS,
        "k": K,
        "n_perturbations": len(perts),
        "rows": rows,
        "summary": summary,
        "evidence_boundary": (
            "Norman2019 only; scVI and PCA were trained/fitted on control cells "
            "only; target perturbation labels were used only for held-out scoring."),
    }
    print("\nsummary:", summary)
    qlib.save_result("e39_architecture_diversity", out)
    print("E39_DONE")


if __name__ == "__main__":
    main()
