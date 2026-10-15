"""E30: does a transplanted effect vector STEER, or does it merely predict?

THE HYPOTHESIS BEING TESTED (user's, stated precisely).
n2/E3 established that the displacement a perturbation causes in one encoder can
be predicted in a second encoder that has zero perturbation labels. The open
question is whether that displacement is usable as an INTERVENTION:

    is  z_control + delta_p  a valid representation of "this cell, perturbed",
    or merely a point whose coordinates happen to match a measured centroid?

Nothing measured so far can answer this, because nothing in the pipeline ever
CONSUMES a steered vector. Every score to date is a cosine between two vectors.
This adds a consumer.

THE CONSUMER. A linear decoder from gf2 embeddings back to log-CPM expression,
fitted on CONTROL CELLS ONLY. Fitting on controls is not a convenience, it is
the point: it mirrors a model that has never seen perturbation data, which is
the deployment setting the hypothesis is about. The decoder cannot have learned
what perturbations look like because it was never shown one.

THE TEST. For each perturbation p and each control cell embedding z:

    steered      = z + delta_p            (delta_p from DONOR encoders only)
    decode both  z and steered
    predicted DE = decode(steered) - decode(z)
    true DE      = measured mean log-CPM(p cells) - mean log-CPM(control cells)
    score        = cosine(predicted DE, true DE) over HVGs

ARMS, each isolating one alternative explanation:

    transplant   delta_p fitted in scGPT + gf1 only. ZERO gf2 perturbation
                 labels. This is the claim.
    native       delta_p fitted in gf2 itself. Ceiling for the steering path:
                 how well this works when the displacement is measured rather
                 than transferred.
    generic      mean delta over all OTHER perturbations. Controls for "any
                 treatment direction decodes to something DE-like".
    random       a random vector with the same norm as delta_p. Floor.
    identity     no displacement. Must score ~0 by construction.

    real_cells   decode ACTUAL perturbed cell embeddings instead of steering.
                 THE CRITICAL CONTROL. It bounds what the decoder can express
                 at all. If real cells decode poorly, the decoder is the
                 bottleneck and a low steering score says nothing about the
                 steering hypothesis.

LEAK CONTROL. Donor deltas are built from the FITTING HALF of each condition's
cells (LESSONS rule 16: all encoders embed the same physical cells in the same
row order, so an encoder index is not a sample index). The true DE target is
built from the disjoint half. The decoder is fitted on control cells, which are
disjoint from every perturbed condition.

INTERPRETATION.
    transplant ~ native, both >> generic   -> the hypothesis holds: steering
                                              works and transfers
    native good, transplant ~ generic      -> displacement transfers as a
                                              coordinate but not as an
                                              intervention
    native ~ generic, real_cells good      -> steering itself fails; the
                                              additive-displacement model of a
                                              perturbation is wrong in this space
    real_cells poor                        -> decoder-limited, test inconclusive
"""
import numpy as np
import qlib

TARGET = "gf2"
DONORS = ["scgpt", "gf1"]
K = 24
N_HVG = 2000
MIN_CELLS = 60


def main():
    idx, syms, _ = qlib.condition_index()
    C = qlib.CONTROL
    models = DONORS + [TARGET]

    # ---- shared space on control cells only (identical to n2/E3) ----------
    betas, mus, views = {}, {}, {}
    for m in models:
        Z, lib, ng = qlib.load_emb(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m], mus[m], views[m] = b, Zd.mean(0), Zd - mus.get(m, Zd.mean(0))
    # recompute cleanly (the line above referenced mus mid-build)
    for m in models:
        Z, lib, ng = qlib.load_emb(C, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m])
        mus[m] = Zd.mean(0)
        views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)

    def S_of(cond, m):
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ g["P"][m]) @ g["W"][m] @ g["C"][m]

    # ---- decoder: shared-space coords -> expression, CONTROLS ONLY --------
    # Decoding from the shared space (not raw gf2) keeps the steering and the
    # decoding in the same coordinate system, which is what the hypothesis is
    # about. The decoder never sees a perturbed cell.
    Xc = qlib.load_rows(idx[C]).astype(np.float64)
    libc = np.maximum(Xc.sum(1, keepdims=True), 1)
    Xcn = np.log1p(Xc / libc * 1e4)
    det = (Xc > 0).mean(0)
    ok = np.flatnonzero(det >= 0.05)
    hvg = ok[np.argsort(-Xcn[:, ok].var(0))[:N_HVG]]
    Ycn = Xcn[:, hvg]

    Sc = S_of(C, TARGET)
    n = min(len(Sc), len(Ycn))
    Sc, Ycn = Sc[:n], Ycn[:n]
    W_dec, alpha = qlib.ridge_map(Sc - Sc.mean(0), Ycn - Ycn.mean(0))
    # held-out decoder quality, so we know the consumer works at all
    hh = n // 2
    Wtr, _ = qlib.ridge_map(Sc[:hh] - Sc[:hh].mean(0), Ycn[:hh] - Ycn[:hh].mean(0))
    pred = (Sc[hh:] - Sc[:hh].mean(0)) @ Wtr + Ycn[:hh].mean(0)
    r_dec = float(np.mean([qlib.cos(pred[i] - Ycn[:hh].mean(0),
                                    Ycn[hh:][i] - Ycn[:hh].mean(0))
                           for i in range(len(pred))]))
    print(f"decoder: {N_HVG} HVGs, alpha={alpha:.1f}, "
          f"held-out per-cell cos={r_dec:.3f}", flush=True)

    perts = [c for c in sorted(idx.keys() - {C}) if len(idx[c]) >= MIN_CELLS]
    print(f"{len(perts)} perturbations, target={TARGET}, donors={DONORS}",
          flush=True)

    S0 = {m: S_of(C, m) for m in models}
    Sp = {p: {m: S_of(p, m) for m in models} for p in perts}

    generic = {m: np.mean([Sp[p][m].mean(0) - S0[m].mean(0) for p in perts],
                          axis=0) for m in models}

    dec = lambda s: (s - Sc.mean(0)) @ W_dec + Ycn.mean(0)
    z_ctrl = Sc.mean(0)                      # a representative control state
    base = dec(z_ctrl)

    rng = np.random.default_rng(0)
    rows = []
    for p in perts:
        # ---- true DE from the HELD-OUT half of this condition's cells ------
        Xp = qlib.load_rows(idx[p]).astype(np.float64)
        libp = np.maximum(Xp.sum(1, keepdims=True), 1)
        Xpn = np.log1p(Xp / libp * 1e4)[:, hvg]
        hp = len(Xpn) // 2
        true_de = Xpn[hp:].mean(0) - Ycn.mean(0)

        h = len(Sp[p][TARGET]) // 2          # fitting half, disjoint from gold
        d_tp = np.mean([Sp[p][m][:h].mean(0) - S0[m].mean(0) for m in DONORS],
                       axis=0)
        d_na = Sp[p][TARGET][:h].mean(0) - S0[TARGET].mean(0)
        d_ge = generic[TARGET]
        d_rd = rng.normal(size=len(d_tp))
        d_rd *= np.linalg.norm(d_tp) / np.linalg.norm(d_rd)

        r = {"pert": p}
        for name, d in [("transplant", d_tp), ("native", d_na),
                        ("generic", d_ge), ("random", d_rd),
                        ("identity", np.zeros_like(d_tp))]:
            r[name] = qlib.cos(dec(z_ctrl + d) - base, true_de)
        # decode REAL perturbed embeddings: what the decoder can express at all
        r["real_cells"] = qlib.cos(
            dec(Sp[p][TARGET][hp:].mean(0)) - base, true_de)
        rows.append(r)

    ARMS = ["real_cells", "native", "transplant", "generic", "random", "identity"]
    print(f"\n{'arm':<12} {'mean cos':>9} {'sd':>7}   vs true differential expression")
    out = {"target": TARGET, "donors": DONORS, "n_perts": len(rows),
           "decoder_heldout_cos": r_dec, "arms": {}}
    for a in ARMS:
        v = np.array([r[a] for r in rows])
        out["arms"][a] = {"mean": float(v.mean()), "sd": float(v.std())}
        print(f"{a:<12} {v.mean():9.3f} {v.std():7.3f}", flush=True)

    def paired(a, b):
        d = np.array([r[a] for r in rows]) - np.array([r[b] for r in rows])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"delta": float(d.mean()), "ci": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0)}

    print("\npaired comparisons (95% bootstrap):")
    for a, b in [("transplant", "generic"), ("native", "generic"),
                 ("transplant", "native"), ("transplant", "random"),
                 ("real_cells", "native")]:
        c = paired(a, b)
        out["arms"][f"{a}_minus_{b}"] = c
        star = "*" if c["significant"] else " "
        print(f"  {a:<11} - {b:<11} {c['delta']:+.4f} "
              f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]{star}", flush=True)

    tp = out["arms"]["transplant"]["mean"]
    ge = out["arms"]["generic"]["mean"]
    rc = out["arms"]["real_cells"]["mean"]
    na = out["arms"]["native"]["mean"]
    if rc < 0.15:
        v = "INCONCLUSIVE: decoder cannot express perturbed states; not a test of steering"
    elif out["arms"]["transplant_minus_generic"]["significant"] and tp > ge:
        v = ("POSITIVE: steering with a transplanted delta produces the correct "
             "differential expression in a decoder that never saw a perturbation")
    elif na > ge:
        v = "PARTIAL: native steering works, transplanted does not transfer as an intervention"
    else:
        v = "NEGATIVE: additive displacement does not steer in this space"
    out["verdict"] = v
    print(f"\nVERDICT: {v}", flush=True)
    qlib.save_result("e30_steering_decode", out)
    print("E30_DONE", flush=True)


if __name__ == "__main__":
    main()
