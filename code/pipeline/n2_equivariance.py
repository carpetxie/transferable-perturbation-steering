"""New-plan Experiment 2: are perturbations coordinate-invariant OPERATORS?

THE CLAIM BEING TESTED, and why it is stronger than vector transfer.

Ordinary cross-model transfer asks whether a STATE maps across coordinates:
    C_A f_A(x) ~ C_B f_B(x)
That is already known to be possible in language models and is not novel.

This asks whether a TRANSFORMATION is preserved. Writing F_p for the action of
perturbation p in the shared space S, the commutative-diagram condition is

    C_m f_m(x^p)  ~  F_p( C_m f_m(x^0) )      for EVERY model m

that is: perturb-then-change-coordinates must agree with change-coordinates-
then-apply-the-shared-operator. A single F_p is fitted once and must work in
every model's coordinates. That is an equivariance claim about interventions,
not a similarity claim about representations.

OPERATOR HIERARCHY, fitted in this order, because jumping to a nonlinear field
would hide which assumption actually fails:

    M0  F_p(s) = s + b_p                       global additive vector (scGen-like)
    M1  F_p(s) = s + b_p + U_p V_p^T (s - mu)  low-rank affine: response depends
                                               linearly on starting state
    M2  context-gated affine: M1 with the low-rank term gated by soft assignment
        to K cell-state contexts (here: cell-cycle-derived), so the operator may
        act differently depending on cell state

THE DECISIVE EVALUATION IS LEAVE-ONE-MODEL-OUT. The operator is fitted using
only models != m, then applied in model m's coordinates and scored against
model m's measured perturbed state. A shared operator that matches a NATIVELY
fitted operator under this split is the strong result. This is the test that
cannot be passed by mere representational similarity.

CONTROLS
    native      F_p fitted and evaluated in model m itself (the ceiling)
    exprPCA     the same operator hierarchy in expression-PCA coordinates
    randbasis   the identical pipeline with a RANDOM K-dim projection of each
                model's raw space in place of the learned GCCA basis. (An
                earlier version rotated the shared space instead; that is an
                orthogonal transform, ridge is nearly rotation-invariant, so it
                reproduced M1 exactly and could not fail.)
    shuffled    perturbation labels deranged (the floor)
    identity    F_p(s) = s, i.e. predict no change (the do-nothing baseline)

DISTRIBUTIONAL SCORING. Perturbations move populations, not just centroids, so
each operator is scored on held-out CELLS by:
    centroid cosine   direction of the mean displacement
    energy distance   between predicted and observed perturbed populations
Centroid-only scoring would credit an operator that gets the mean right and the
spread wrong.
"""
import numpy as np
import qlib

MODELS = ["scgpt", "gf1", "gf2"]
K = 24
N_CTX = 3          # contexts for M2


def energy_distance(X, Y, rng, n=120):
    """Energy distance between two point clouds, subsampled for tractability."""
    if len(X) > n:
        X = X[rng.permutation(len(X))[:n]]
    if len(Y) > n:
        Y = Y[rng.permutation(len(Y))[:n]]
    def pd(A, B):
        return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0)).mean()
    return float(2 * pd(X, Y) - pd(X, X) - pd(Y, Y))


def fit_M0(S0, S1):
    """Additive: the mean displacement."""
    return {"b": (S1.mean(0) - S0.mean(0))}


def apply_M0(op, S):
    return S + op["b"]


def fit_M1(S0, S1, rank=6, alpha=1.0):
    """Low-rank affine: s -> s + b + UV^T (s - mu), fitted by ridge then SVD."""
    mu = S0.mean(0)
    b = S1.mean(0) - mu
    X = S0 - mu
    Y = (S1 - S0) - b                      # residual after the additive part
    W = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(rank, len(s))
    return {"b": b, "mu": mu, "W": (U[:, :r] * s[:r]) @ Vt[:r]}


def apply_M1(op, S):
    return S + op["b"] + (S - op["mu"]) @ op["W"]


def fit_M2(S0, S1, ctx0, rank=4, alpha=1.0):
    """Context-gated affine: one low-rank affine operator per cell-state context."""
    ops = {}
    for k in range(N_CTX):
        m = ctx0 == k
        if m.sum() < 20:
            ops[k] = None
            continue
        ops[k] = fit_M1(S0[m], S1[m], rank=rank, alpha=alpha)
    glob = fit_M1(S0, S1, rank=rank, alpha=alpha)
    return {"ops": ops, "glob": glob}


def apply_M2(op, S, ctx):
    out = np.empty_like(S)
    for k in range(N_CTX):
        m = ctx == k
        if not m.any():
            continue
        o = op["ops"].get(k) or op["glob"]
        out[m] = apply_M1(o, S[m])
    return out


def main():
    idx, syms, triples = qlib.condition_index()
    rng = np.random.default_rng(0)

    # ---- ONE shared space for all models, via generalized CCA on controls.
    # Pairwise per-model CCA would give each model a basis fitted against a
    # different partner, and an operator fitted in one such basis would not be
    # applicable in another. Equivariance is only testable in a common space.
    betas, mus, views = {}, {}, {}
    for m in MODELS:
        Z, lib, ng = qlib.load_emb("control", m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b
        mus[m] = Zd.mean(0)
        views[m] = Zd - mus[m]
    gcca = qlib.fit_gcca(views, k=K)

    def S_of(cond, m):
        """Per-CELL shared coordinates for one condition in model m."""
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ gcca["P"][m]) @ gcca["W"][m] @ gcca["C"][m]

    # cell-state context from cell cycle, computed once in expression space
    from q5_coordinates import cycle_scores
    cyc_ctrl = cycle_scores(idx["control"], syms)
    edges = np.quantile(cyc_ctrl, [1 / 3, 2 / 3])
    ctx_of = lambda v: np.digitize(v, edges)

    S0 = {m: S_of("control", m) for m in MODELS}
    ctx0 = ctx_of(cyc_ctrl)

    # perturbations with enough cells to fit and score an operator
    perts = [c for c in sorted(idx.keys() - {"control"}) if len(idx[c]) >= 60]
    print(f"{len(perts)} perturbations, {len(MODELS)} models, K={K}", flush=True)

    # CONTROLS. An earlier version rotated the shared space, fitted there, and
    # rotated back. That reproduces the same affine operator (ridge is nearly
    # rotation-invariant), scored identically to M1, and could not fail. It is
    # replaced by a RANDOM-BASIS control: the identical pipeline with a random
    # K-dim projection of each model's raw space substituted for the learned
    # GCCA basis. If the operator transfers just as well there, the shared basis
    # is doing no work.
    Qrand = {m: np.linalg.qr(rng.normal(size=(views[m].shape[1], K)))[0]
             for m in MODELS}

    def S_rand(cond, m):
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return Zd @ Qrand[m]

    S0r = {m: S_rand("control", m) for m in MODELS}
    R = np.linalg.qr(rng.normal(size=(K, K)))[0]
    shuf = dict(zip(perts, rng.permutation(perts)))

    rows = []
    for p in perts:
        Sp = {m: S_of(p, m) for m in MODELS}
        cyc_p = cycle_scores(idx[p], syms)
        ctxp = ctx_of(cyc_p)
        # control cells are shared across models; subsample to match p's count
        for m_out in MODELS:
            m_in = [m for m in MODELS if m != m_out]
            # ---- fit each operator family on the OTHER models, pooled
            fitS0 = np.vstack([S0[m] for m in m_in])
            fitS1 = np.vstack([Sp[m] for m in m_in])
            # pair control cells to perturbed cells by index recycling
            n = min(len(fitS0), len(fitS1))
            A0, A1 = fitS0[:n], fitS1[:n]
            # SAMPLE-MATCHED cross-model fit. The shared operator pools cells
            # from two models while the native ceiling is fitted on half of one
            # model's cells. Comparing them directly rewards the shared arm for
            # a larger data budget rather than for portability, so a matched
            # variant is fitted on the SAME number of cells the native arm gets.
            n_match = max(10, min(len(S0[m_out]), len(Sp[m_out])) // 2)
            A0m, A1m = fitS0[:n_match], fitS1[:n_match]
            ops = {"M0": (fit_M0(A0, A1), apply_M0),
                   "M1": (fit_M1(A0, A1), apply_M1),
                   "M1match": (fit_M1(A0m, A1m), apply_M1)}
            # evaluate in the held-out model's coordinates
            E0, E1 = S0[m_out], Sp[m_out]
            n2 = min(len(E0), len(E1))
            E0s, E1s = E0[:n2], E1[:n2]
            true_c = E1s.mean(0) - E0s.mean(0)
            r = {"pert": p, "model_out": m_out}
            for nm, (op, ap) in ops.items():
                pred = ap(op, E0s)
                r[f"{nm}_cos"] = qlib.cos(pred.mean(0) - E0s.mean(0), true_c)
                r[f"{nm}_energy"] = energy_distance(pred, E1s, rng)
            # NATIVE CEILING, honestly split. Fitting and evaluating on the
            # same cells makes this trivially 1.000: fit_M1 sets b to the mean
            # displacement, so the predicted centroid reproduces the observed
            # one by construction. That is not a ceiling, it is an identity.
            # Fit on one half of the held-out model's cells, score on the other.
            h = len(E0s) // 2
            if h >= 10:
                r["_h"] = h
                nat = fit_M1(E0s[:h], E1s[:h])
                pn = apply_M1(nat, E0s[h:])
                tc_h = E1s[h:].mean(0) - E0s[h:].mean(0)
                r["native_cos"] = qlib.cos(pn.mean(0) - E0s[h:].mean(0), tc_h)
                r["native_energy"] = energy_distance(pn, E1s[h:], rng)
                # split-half reliability of the target itself: the highest
                # cosine any method could achieve given sampling noise
                r["noise_ceiling_cos"] = qlib.cos(
                    E1s[:h].mean(0) - E0s[:h].mean(0), tc_h)
                # Score the sample-matched SHARED operator on the SAME held-out
                # half and against the SAME target as native. Scoring it against
                # the full-sample target instead would give it a less noisy
                # goal than native gets, which is not a fair comparison.
                pm = apply_M1(ops["M1match"][0], E0s[h:])
                r["M1match_fair_cos"] = qlib.cos(
                    pm.mean(0) - E0s[h:].mean(0), tc_h)
            else:
                r["native_cos"] = float("nan")
                r["native_energy"] = float("nan")
                r["noise_ceiling_cos"] = float("nan")
            # identity baseline
            r["identity_cos"] = 0.0
            r["identity_energy"] = energy_distance(E0s, E1s, rng)
            # random-basis control: same pipeline, learned basis -> random basis
            Spr = {m: S_rand(p, m) for m in MODELS}
            f0r = np.vstack([S0r[m] for m in m_in])
            f1r = np.vstack([Spr[m] for m in m_in])
            nr = min(len(f0r), len(f1r))
            oprb = fit_M1(f0r[:nr], f1r[:nr])
            E0r, E1r = S0r[m_out], Spr[m_out]
            n2r = min(len(E0r), len(E1r))
            prb = apply_M1(oprb, E0r[:n2r])
            r["randbasis_cos"] = qlib.cos(prb.mean(0) - E0r[:n2r].mean(0),
                                          E1r[:n2r].mean(0) - E0r[:n2r].mean(0))
            # shuffled-label control on M0
            Ssh = {m: S_of(shuf[p], m) for m in m_in}
            osh = fit_M0(np.vstack([S0[m] for m in m_in])[:n],
                         np.vstack([Ssh[m] for m in m_in])[:n])
            r["shuffled_cos"] = qlib.cos(
                apply_M0(osh, E0s).mean(0) - E0s.mean(0), true_c)
            rows.append(r)

    def mean(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    print(f"\nleave-one-model-out, {len(rows)} (perturbation, held-out model) cells")
    print(f"{'operator':<12} {'centroid cos':>13} {'energy dist':>13}")
    for nm in ["M0", "M1", "M1match"]:
        print(f"{nm:<12} {mean(nm+'_cos'):13.3f} {mean(nm+'_energy'):13.3f}")
    print(f"{'native(M1)':<12} {mean('native_cos'):13.3f} {mean('native_energy'):13.3f}"
          "   (split-half fit, not in-sample)")
    print(f"{'noise ceil':<12} {mean('noise_ceiling_cos'):13.3f}"
          "                 (split-half reliability of the target)")
    print(f"{'identity':<12} {mean('identity_cos'):13.3f} {mean('identity_energy'):13.3f}")
    print(f"{'randbasis':<12} {mean('randbasis_cos'):13.3f}")
    print(f"{'shuffled':<12} {mean('shuffled_cos'):13.3f}")

    gap = mean("native_cos") - mean("M1match_fair_cos")
    print(f"\nSAMPLE-MATCHED comparison (both fitted on the same cell budget):")
    print(f"  shared {mean('M1match_fair_cos'):.3f}  vs  native "
          f"{mean('native_cos'):.3f}  vs  noise ceiling "
          f"{mean('noise_ceiling_cos'):.3f}")
    print("  (same cell budget, same held-out half, same target)")
    print(f"M1 - M0 (does state-dependence help?): "
          f"{mean('M1_cos') - mean('M0_cos'):+.3f}")
    print(f"M1 - randbasis (is the LEARNED shared basis doing the work?): "
          f"{mean('M1_cos') - mean('randbasis_cos'):+.3f}")
    print(f"\nequivariance gap (native - shared M1): {gap:+.3f} "
          f"(near 0 = shared operator matches a natively fitted one)")
    qlib.save_result("n2_equivariance", {
        "K": K, "models": MODELS, "n_cells": len(rows),
        "summary": {k: mean(k) for k in
                    ["M0_cos", "M1_cos", "M1match_cos", "M1match_fair_cos",
                     "native_cos", "identity_cos",
                     "randbasis_cos", "shuffled_cos", "noise_ceiling_cos", "M0_energy", "M1_energy",
                     "native_energy", "identity_energy"]},
        "equivariance_gap": gap, "per_row": rows})
    print("N2_DONE", flush=True)


if __name__ == "__main__":
    main()
