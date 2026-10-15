"""E31: is E30's positive real, or is it generic treatment structure?

E30 reported that steering a control state with a transplanted delta produces
the correct differential expression (transplant 0.575, generic 0.470,
real_cells 0.592). Three features of that output demand verification before it
is believed, and this project has already retracted three results that looked
this good.

FLAG 1. The generic arm scores 0.470 of a 0.592 ceiling. Most of the raw score
is the direction shared by ALL perturbations. A method could look strong while
carrying almost no perturbation-specific information.

FLAG 2. transplant - native = -0.0016, essentially identical, where the same
comparison in embedding space differed by -0.034. The decoder is rank <= 24, so
it may be washing out exactly the difference the comparison is meant to detect.

FLAG 3. The decoder's ridge alpha selected the smallest value in its grid (a
grid-edge selection) and its held-out per-cell cosine is 0.206, which is weak.

THE CONTROLS ADDED HERE.

  mismatch      steer with the delta of a DIFFERENT, randomly chosen
                perturbation. THE DECISIVE TEST. If a wrong delta decodes as
                well as the right one, the pipeline is only reproducing generic
                treatment structure and E30's headline is empty. The correct
                delta must beat its own mismatch, paired per perturbation.

  specific      project the generic direction OUT of both the predicted and the
                true differential expression, then score. This removes the
                shared component entirely rather than merely reporting it as a
                separate arm, which is the Systema requirement.

  alpha sweep   re-fit the decoder across a wide alpha grid and report whether
                the ranking of arms is stable, so the grid-edge selection cannot
                silently drive the result.

  rank sweep    vary the shared-space dimension K. If transplant only equals
                native because the decoder is too low-rank to tell them apart,
                the gap should reappear at larger K.

INTERPRETATION.
  correct >> mismatch, and specific gain survives  -> E30 stands
  correct ~ mismatch                               -> E30 is generic structure
  gap reappears at larger K                        -> E30's transplant~native
                                                      was a rank artifact
"""
import numpy as np
import qlib

TARGET = "gf2"
DONORS = ["scgpt", "gf1"]
N_HVG = 2000
MIN_CELLS = 60
KS = [24, 48]
ALPHAS = np.logspace(-2, 5, 15)


def build(K):
    idx, syms, _ = qlib.condition_index()
    C = qlib.CONTROL
    models = DONORS + [TARGET]
    betas, mus, views = {}, {}, {}
    for m in models:
        Z, lib, ng = qlib.load_emb(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)

    def S_of(cond, m):
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ g["P"][m]) @ g["W"][m] @ g["C"][m]
    return idx, C, models, S_of


def main():
    out = {"target": TARGET, "donors": DONORS, "K_sweep": {}}
    rng = np.random.default_rng(0)

    for K in KS:
        idx, C, models, S_of = build(K)

        Xc = qlib.load_rows(idx[C]).astype(np.float64)
        Xcn = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4)
        det = (Xc > 0).mean(0)
        ok = np.flatnonzero(det >= 0.05)
        hvg = ok[np.argsort(-Xcn[:, ok].var(0))[:N_HVG]]
        Ycn = Xcn[:, hvg]
        Sc = S_of(C, TARGET)
        n = min(len(Sc), len(Ycn)); Sc, Ycn = Sc[:n], Ycn[:n]

        perts = [c for c in sorted(idx.keys() - {C}) if len(idx[c]) >= MIN_CELLS]
        S0 = {m: S_of(C, m) for m in models}
        Sp = {p: {m: S_of(p, m) for m in models} for p in perts}
        generic_t = np.mean([Sp[p][TARGET].mean(0) - S0[TARGET].mean(0)
                             for p in perts], axis=0)

        # true DE per perturbation, from the held-out half
        trueDE = {}
        for p in perts:
            Xp = qlib.load_rows(idx[p]).astype(np.float64)
            Xpn = np.log1p(Xp / np.maximum(Xp.sum(1, keepdims=True), 1)*1e4)[:, hvg]
            trueDE[p] = Xpn[len(Xpn)//2:].mean(0) - Ycn.mean(0)
        gen_de = np.mean([trueDE[p] for p in perts], axis=0)   # generic DE axis

        def rm_generic(v):
            u = gen_de / np.linalg.norm(gen_de)
            return v - (v @ u) * u

        for alpha in ([None] if K != KS[0] else [None, 1.0, 100.0]):
            if alpha is None:
                W_dec, a_used = qlib.ridge_map(Sc - Sc.mean(0),
                                               Ycn - Ycn.mean(0), alphas=ALPHAS)
            else:
                G = (Sc - Sc.mean(0))
                W_dec = np.linalg.solve(G.T@G + alpha*np.eye(G.shape[1]),
                                        G.T @ (Ycn - Ycn.mean(0)))
                a_used = alpha
            dec = lambda s: (s - Sc.mean(0)) @ W_dec + Ycn.mean(0)
            z0 = Sc.mean(0); base = dec(z0)

            rows = []
            for i, p in enumerate(perts):
                h = len(Sp[p][TARGET]) // 2
                d_tp = np.mean([Sp[p][m][:h].mean(0) - S0[m].mean(0)
                                for m in DONORS], axis=0)
                d_na = Sp[p][TARGET][:h].mean(0) - S0[TARGET].mean(0)
                # mismatch: a DIFFERENT perturbation's transplanted delta
                q = perts[int(rng.integers(len(perts)))]
                while q == p:
                    q = perts[int(rng.integers(len(perts)))]
                hq = len(Sp[q][TARGET]) // 2
                d_mm = np.mean([Sp[q][m][:hq].mean(0) - S0[m].mean(0)
                                for m in DONORS], axis=0)
                t = trueDE[p]
                r = {"pert": p}
                for nm, d in [("transplant", d_tp), ("native", d_na),
                              ("mismatch", d_mm), ("generic", generic_t)]:
                    pred = dec(z0 + d) - base
                    r[nm] = qlib.cos(pred, t)
                    r[nm + "_spec"] = qlib.cos(rm_generic(pred), rm_generic(t))
                rows.append(r)

            def paired(a, b):
                d = np.array([r[a] for r in rows]) - np.array([r[b] for r in rows])
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                return {"delta": float(d.mean()), "ci": [float(lo), float(hi)],
                        "significant": bool(lo > 0 or hi < 0)}

            tag = f"K{K}_alpha{a_used:g}"
            res = {"alpha": float(a_used), "K": K, "n": len(rows)}
            for nm in ["transplant", "native", "mismatch", "generic"]:
                res[nm] = float(np.mean([r[nm] for r in rows]))
                res[nm + "_spec"] = float(np.mean([r[nm + "_spec"] for r in rows]))
            res["transplant_minus_mismatch"] = paired("transplant", "mismatch")
            res["transplant_minus_mismatch_spec"] = paired("transplant_spec",
                                                           "mismatch_spec")
            res["transplant_minus_native"] = paired("transplant", "native")
            out["K_sweep"][tag] = res

            print(f"\n=== {tag}  (n={len(rows)}) ===", flush=True)
            print(f"{'arm':<12} {'raw cos':>9} {'generic-removed':>17}")
            for nm in ["transplant", "native", "mismatch", "generic"]:
                print(f"{nm:<12} {res[nm]:9.3f} {res[nm+'_spec']:17.3f}", flush=True)
            m1 = res["transplant_minus_mismatch"]
            m2 = res["transplant_minus_mismatch_spec"]
            m3 = res["transplant_minus_native"]
            print(f"  transplant - mismatch  raw {m1['delta']:+.4f} "
                  f"[{m1['ci'][0]:+.4f},{m1['ci'][1]:+.4f}]"
                  f"{'*' if m1['significant'] else ' '}", flush=True)
            print(f"  transplant - mismatch  spec {m2['delta']:+.4f} "
                  f"[{m2['ci'][0]:+.4f},{m2['ci'][1]:+.4f}]"
                  f"{'*' if m2['significant'] else ' '}", flush=True)
            print(f"  transplant - native    raw {m3['delta']:+.4f} "
                  f"[{m3['ci'][0]:+.4f},{m3['ci'][1]:+.4f}]"
                  f"{'*' if m3['significant'] else ' '}", flush=True)

    key = [v for k, v in out["K_sweep"].items() if k.startswith(f"K{KS[0]}_")][0]
    mm = key["transplant_minus_mismatch_spec"]
    if mm["significant"] and mm["delta"] > 0:
        v = ("HOLDS: the correct delta beats a mismatched one even after the "
             "generic direction is removed")
    else:
        v = ("FAILS: a mismatched delta decodes as well as the correct one; "
             "E30 was generic treatment structure")
    out["verdict"] = v
    print(f"\nVERDICT: {v}", flush=True)
    qlib.save_result("e31_steering_hardening", out)
    print("E31_DONE", flush=True)


if __name__ == "__main__":
    main()
