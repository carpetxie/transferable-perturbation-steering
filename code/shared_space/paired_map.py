"""A linear map that is actually a fair baseline.

THE DEFECT THIS REPLACES
------------------------
`sweep.test_pair` fitted the linear arm with `np.linalg.lstsq(..., rcond=None)`
on 800 training cells. Embedding spaces here are 256 to 1152 dimensional, so
whenever d_source approaches or exceeds n_train the system is underdetermined
and lstsq returns a minimum-norm solution that generalises terribly. Measured on
PBMC, same cells, same splits:

    pair          d_src   lstsq   ridge(100)
    gf1->gf2        256   0.407     0.415      <- unaffected, d << n
    scgpt->gf2      512   0.630     0.728
    gf2->gf3        768   0.407     0.963
    gf2->pca        768   0.075     0.940      <- 12x artifact
    gf3->pca       1152   0.700     0.978

The MLP arm has early stopping and an L2 penalty, so it was regularised while
the linear arm was not. Every 'linear vs MLP' number in Finding 4 was therefore
comparing a crippled linear baseline against a regularised nonlinear one, and
the size of the handicap grew with source dimension.

THE FIX
-------
Ridge with alpha chosen by cross-validation INSIDE the training split. The
held-out cells never influence alpha, so this is not tuning on the test set. The
alpha grid is wide (1e-2 to 1e4) and the chosen value is reported, so a fit that
pins to a grid edge is visible rather than silent.
"""

from __future__ import annotations

import numpy as np


def rank_metrics(P, Z):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Pn @ Zn.T
    order = np.argsort(-S, axis=1)
    r = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(len(S))])
    return float((r == 1).mean()), float(r.mean())


ALPHAS = np.logspace(-2, 4, 13)


def _ridge_cv(A, B, alphas=ALPHAS, folds=4, seed=0):
    """Pick alpha by k-fold CV within the training set only."""
    n = len(A)
    idx = np.random.default_rng(seed).permutation(n)
    cuts = np.array_split(idx, folds)
    err = np.zeros(len(alphas))
    for f in range(folds):
        va = cuts[f]
        tr = np.concatenate([cuts[g] for g in range(folds) if g != f])
        At, Bt = A[tr], B[tr]
        G = At.T @ At
        R = At.T @ Bt
        for j, al in enumerate(alphas):
            W = np.linalg.solve(G + al * np.eye(G.shape[0]), R)
            err[j] += float(((A[va] @ W - B[va]) ** 2).mean())
    best = alphas[int(np.argmin(err))]
    G = A.T @ A
    W = np.linalg.solve(G + best * np.eye(G.shape[0]), A.T @ B)
    return W, float(best)


def test_pair(zA, zB, seed=0, with_mlp=True):
    """Paired ridge (CV-tuned) vs paired MLP on held-out cells."""
    zA = (zA - zA.mean(0)) / (zA.std(0) + 1e-6)
    zB = (zB - zB.mean(0)) / (zB.std(0) + 1e-6)
    n = len(zA)
    p = np.random.default_rng(seed).permutation(n)
    tr, ev = p[: int(0.8 * n)], p[int(0.8 * n):]

    W, alpha = _ridge_cv(zA[tr], zB[tr], seed=seed)
    lt, lr = rank_metrics(zA[ev] @ W, zB[ev])

    out = {"n_eval": len(ev), "chance_top1": 1 / len(ev), "d_src": zA.shape[1],
           "linear_top1": lt, "linear_rank": lr, "alpha": alpha,
           "alpha_at_grid_edge": bool(alpha <= ALPHAS[0] or alpha >= ALPHAS[-1])}
    if with_mlp:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(zA[tr])
        mlp = MLPRegressor(hidden_layer_sizes=(512, 512), max_iter=600,
                           early_stopping=True, random_state=seed
                           ).fit(sc.transform(zA[tr]), zB[tr])
        nt, nr = rank_metrics(mlp.predict(sc.transform(zA[ev])), zB[ev])
        out.update({"mlp_top1": nt, "mlp_rank": nr, "linear_wins": bool(lt >= nt)})
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from sweep import embed_all
    from pca_space import get_pca

    print("VALIDATION of the ridge-CV paired map\n")
    E = embed_all("pbmc", 1000)
    S = {k: E[k].astype(np.float64) for k in ("scgpt", "gf1", "gf2", "gf3")}
    S["pca"] = get_pca("pbmc", 1000).astype(np.float64)

    # 1. identity must be near-perfect
    r = test_pair(S["gf2"], S["gf2"].copy(), with_mlp=False)
    print(f"  1. gf2 -> itself        top-1 {r['linear_top1']:.3f}  (must be ~1.0)")

    # 2. destroyed correspondence must fall to chance
    rng = np.random.default_rng(0)
    r2 = test_pair(S["gf2"], S["gf2"][rng.permutation(len(S["gf2"]))], with_mlp=False)
    print(f"  2. gf2 -> shuffled gf2  top-1 {r2['linear_top1']:.3f}  "
          f"(chance {r2['chance_top1']:.3f})")

    # 3. alpha must not sit at a grid edge, which would mean the grid is wrong
    print("\n  3. chosen alpha per pair (grid 1e-2 .. 1e4):")
    edge = False
    for a in S:
        for b in S:
            if a >= b:
                continue
            rr = test_pair(S[a], S[b], with_mlp=False)
            edge |= rr["alpha_at_grid_edge"]
            flag = "  <-- AT GRID EDGE" if rr["alpha_at_grid_edge"] else ""
            print(f"     {a+'->'+b:<14} d_src {rr['d_src']:5d}  alpha {rr['alpha']:9.2f}"
                  f"  top-1 {rr['linear_top1']:.3f}{flag}")

    ok = r["linear_top1"] > 0.95 and r2["linear_top1"] < 5 * r2["chance_top1"] and not edge
    print(f"\n  instrument {'VALID' if ok else 'CHECK FAILED'}")
