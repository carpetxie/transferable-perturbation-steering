"""E3: how many target-model labels does a transplanted perturbation effect save?

THE SETTING, and why it is the product-shaped question. n2 established that a
perturbation effect fitted in two models transfers into a third at 94% of that
third model's own noise ceiling, using a shared space built only from
unperturbed cells. That is a scientific claim. The engineering claim is:

    if you have a NEW frozen encoder and no perturbation labels for it, how much
    labelled data does installing a donor-learned effect save you?

DESIGN. Leave-one-model-out. For held-out model t and perturbation p:

    transplant   b_p fitted in the shared space using ONLY the other two
                 models' cells. Zero perturbation labels from t. The shared
                 basis is fitted on t's UNPERTURBED cells, which is allowed:
                 the whole premise is that a new model can be aligned cheaply
                 with an unlabelled atlas.
    native(k)    b_p fitted natively in t from k of t's own perturbed cells,
                 for k in {3, 5, 10, 20, 40}. This is the curve the transplant
                 has to beat.
    generic      the mean effect over all OTHER perturbations, transplanted.
                 Controls for "any treatment direction would score this",
                 which is the Systema concern in label-budget form.
    identity     no effect applied. The floor.

All arms are scored on the SAME held-out half of t's cells, against the same
target, so no arm gets a less noisy goal than another. The noise ceiling is the
split-half reliability of that target, and it is reported because a method
cannot beat it and a label-savings number is meaningless without it.

DELIVERABLE. The break-even label count: the smallest k at which native(k)
matches the transplant. That is what a lab would actually want to know, because
it says how many wells the transplant is worth.

HONEST LIMIT, stated here because it bounds the claim. Donor and target here are
different ENCODERS applied to the same cells, not different studies or cell
lines. So this measures portability across representations, not across
biological context. The stronger cross-study version needs a second perturbation
dataset and is not run here.
"""
import numpy as np
import qlib

MODELS = ["scgpt", "gf1", "gf2"]
K = 24
KS = [3, 5, 10, 20, 40]
MIN_CELLS = 60


def main():
    idx, syms, triples = qlib.condition_index()
    rng = np.random.default_rng(0)

    betas, mus, views = {}, {}, {}
    for m in MODELS:
        Z, lib, ng = qlib.load_emb("control", m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b
        mus[m] = Zd.mean(0)
        views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)

    def S_of(cond, m):
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ g["P"][m]) @ g["W"][m] @ g["C"][m]

    S0 = {m: S_of("control", m) for m in MODELS}
    perts = [c for c in sorted(idx.keys() - {"control"}) if len(idx[c]) >= MIN_CELLS]
    print(f"{len(perts)} perturbations, {len(MODELS)} models, K={K}", flush=True)

    Sp_cache = {p: {m: S_of(p, m) for m in MODELS} for p in perts}

    # generic effect: mean displacement over all perturbations, per model
    generic = {m: np.mean([Sp_cache[p][m].mean(0) - S0[m].mean(0)
                           for p in perts], axis=0) for m in MODELS}

    rows = []
    for p in perts:
        Sp = Sp_cache[p]
        for t in MODELS:
            donors = [m for m in MODELS if m != t]
            E0, E1 = S0[t], Sp[t]
            n2 = min(len(E0), len(E1))
            E0s, E1s = E0[:n2], E1[:n2]
            h = n2 // 2
            if h < 10:
                continue
            # target: measured effect on the held-out half
            tgt = E1s[h:].mean(0) - E0s[h:].mean(0)
            # noise ceiling: split-half reliability of that target
            ceil = qlib.cos(E1s[:h].mean(0) - E0s[:h].mean(0), tgt)

            # transplant: donors only, zero target perturbation labels.
            # LEAK FIX (see FINDING_09 / LESSONS 16). This previously used ALL
            # donor cells, justified as "the honest product setting". That
            # justification does not hold in THIS measurement: every encoder
            # embeds the same physical cells in the same row order, so donor
            # rows h: are the very cells the gold `tgt` is built from, read
            # through another encoder. Using them scores the transplant partly
            # against itself. In a real deployment the target's evaluation cells
            # would be different cells, so the faithful analogue is to restrict
            # donors to the same fitting half the native arm gets.
            b_tp = np.mean([Sp[m][:h].mean(0) - S0[m].mean(0) for m in donors],
                           axis=0)
            b_gen = np.mean([generic[m] for m in donors], axis=0)

            r = {"pert": p, "target": t, "ceiling": ceil, "h": h,
                 "transplant": qlib.cos(b_tp, tgt),
                 "generic": qlib.cos(b_gen, tgt),
                 "identity": 0.0}
            # native fitted on ALL available target fit-half cells. This, not
            # the split-half reliability, is the reference the transplant must
            # be judged against: it is what you get by labelling this model
            # properly. Reporting "% of ceiling" against the split-half number
            # produced a nonsensical 120%, because the transplant pools ~200
            # donor cells while the split-half estimate uses ~50.
            r["native_full"] = qlib.cos(E1s[:h].mean(0) - E0s[:h].mean(0), tgt)
            for k in KS:
                if k > h:
                    r[f"native{k}"] = np.nan
                    r[f"matched{k}"] = np.nan
                    continue
                sub = rng.permutation(h)[:k]
                r[f"native{k}"] = qlib.cos(
                    E1s[:h][sub].mean(0) - E0s[:h].mean(0), tgt)
                # BUDGET-MATCHED transplant: same total donor cells as native(k)
                per = max(2, k // len(donors))
                b_m = np.mean([Sp[m][:h][rng.permutation(h)[:per]].mean(0)
                               - S0[m].mean(0) for m in donors], axis=0)
                r[f"matched{k}"] = qlib.cos(b_m, tgt)
            rows.append(r)

    def m(k):
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        return float(v.mean()), float(v.std())

    print(f"\nleave-one-model-out, {len(rows)} (perturbation, held-out model) cells")
    print(f"{'arm':<16} {'cos to measured effect':>24}  {'% of native_full':>17}")
    ceil_m = m("native_full")[0]
    out = {"n_rows": len(rows), "ceiling": ceil_m, "arms": {}}
    order = (["native_full", "ceiling", "transplant"]
             + [f"native{k}" for k in KS]
             + [f"matched{k}" for k in KS]
             + ["generic", "identity"])
    for k in order:
        mu, sd = m(k)
        out["arms"][k] = {"mean": mu, "sd": sd, "pct_ceiling": mu / max(ceil_m, 1e-9)}
        print(f"{k:<16} {mu:14.3f} +/- {sd:5.3f}  {mu/max(ceil_m,1e-9):12.0%}")

    # break-even: smallest k whose native mean reaches the transplant mean
    tp = m("transplant")[0]
    be = None
    for k in KS:
        if m(f"native{k}")[0] >= tp:
            be = k
            break
    out["transplant_mean"] = tp
    out["break_even_k"] = be
    print(f"\nBREAK-EVEN: transplant ({tp:.3f}) is matched by native fitting at "
          f"k = {be if be else '>' + str(KS[-1])} target cells per perturbation")

    # paired bootstrap: transplant vs generic, and transplant vs best native<=10
    tpv = np.array([r["transplant"] for r in rows])
    gnv = np.array([r["generic"] for r in rows])
    d = tpv - gnv
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out["transplant_minus_generic"] = {"mean": float(d.mean()),
                                       "ci": [float(lo), float(hi)],
                                       "significant": bool(lo > 0)}
    print(f"transplant - generic: {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
          f"{'SIGNIFICANT' if lo > 0 else 'not significant'}")

    ok = (out["transplant_minus_generic"]["significant"] and
          tp > 0.5 * ceil_m and (be is None or be >= 5))
    verdict = (f"POSITIVE: a transplanted effect using ZERO target labels is worth "
               f"~{be if be else '>'+str(KS[-1])} labelled target cells per "
               f"perturbation, and beats a generic treatment direction"
               if ok else
               "NULL: the transplant does not beat the generic direction or is "
               "matched by a handful of native labels")
    out["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    # paired bootstrap, matched-budget transplant vs native at the SAME k.
    # This is the claim that matters: at equal cell budget, are donor cells
    # from other models a better estimate of THIS model's effect than its own
    # cells? The unmatched comparison cannot answer that.
    print(f"\nmatched-budget transplant vs native, same k, paired bootstrap:")
    out["matched_vs_native"] = {}
    for k in KS:
        a_ = np.array([r[f"matched{k}"] for r in rows], dtype=float)
        b_ = np.array([r[f"native{k}"] for r in rows], dtype=float)
        ok = ~(np.isnan(a_) | np.isnan(b_))
        dd = a_[ok] - b_[ok]
        if len(dd) < 20:
            continue
        bs2 = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
        lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
        out["matched_vs_native"][k] = {"delta": float(dd.mean()),
                                       "ci": [float(lo2), float(hi2)],
                                       "significant": bool(lo2 > 0),
                                       "win_rate": float((dd > 0).mean())}
        print(f"  k={k:3d}  {dd.mean():+.4f} [{lo2:+.4f},{hi2:+.4f}] "
              f"{'SIG' if lo2>0 else '   '}  win {float((dd>0).mean()):.0%}")
    out["rows"] = rows
    qlib.save_result("e3_compiler", out)
    print("E3_DONE", flush=True)


if __name__ == "__main__":
    main()
