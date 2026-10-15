"""E37: does the E34 debiasing result survive closing the cell-overlap leak?

E34/E35/E36 established that donor bias is consistent across perturbations and
that subtracting a cross-fitted mean bias beats both raw donor transfer and
native estimation on all three datasets. But those scripts sampled donor cells
from the FULL per-condition cell list, while the gold target effect was
computed from the second half of the same physical cells. Every encoder embeds
the same cells in the same row order (rule 16), so donor estimates could read
the gold cells through another encoder. That exact mechanism produced the
withdrawn FINDING_09, so the debias result cannot stand until it is re-run
with donors restricted to cells DISJOINT from the gold half.

A second required piece of honesty: e34's debiased cosine exceeded
sqrt(split-half ceiling) (0.79 vs 0.75 on Norman, 0.885 vs 0.76 on Papalexi).
That bound applies to estimators with no information about the target encoder
beyond the test perturbation; the debiased estimator imports the pilot
perturbations' mean target effect (the generic direction, measured with low
noise across ~100 perturbations), so it can legitimately exceed the bound on
raw cosine. The per-perturbation content is what retrieval measures, so this
script reports retrieval too, all arms leak-free.

Both versions are computed side by side:
  leaky    -- donor cells drawn from all rows (E34's design)
  disjoint -- donor cells drawn only from rows [:h]; gold is rows [h:];
              control means for donors exclude the gold control rows
The difference between them IS the measured size of the leak.
"""
import numpy as np, qlib

M = ["scgpt", "gf1", "gf2"]
K = 24
BUDGET = 24


def retrieval(preds, targs):
    n = len(preds)
    P = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in preds])
    T = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in targs])
    Sim = P @ T.T
    order = np.argsort(-Sim, axis=1)
    rk = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
    return float((rk == 1).mean()), float(rk.mean()), n


def main():
    idx, syms, _ = qlib.condition_index()
    rng = np.random.default_rng(0)
    C = qlib.CONTROL
    betas, mus, views = {}, {}, {}
    for m in M:
        Z, lib, ng = qlib.load_emb(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)
    S = lambda c, m: ((qlib.apply_depth_projector(*qlib.load_emb(c, m),
                                                  betas[m]) - mus[m])
                      @ g["P"][m]) @ g["W"][m] @ g["C"][m]
    S0 = {m: S(C, m) for m in M}
    perts = [c for c in sorted(idx.keys() - {C}) if len(idx[c]) >= 60]
    Sp = {p: {m: S(p, m) for m in M} for p in perts}

    rec = []
    for p in perts:
        for t in M:
            donors = [m for m in M if m != t]
            E0, E1 = S0[t], Sp[p][t]
            n2 = min(len(E0), len(E1)); h = n2 // 2
            if h < 10:
                continue
            E0s, E1s = E0[:n2], E1[:n2]
            tgt = E1s[h:].mean(0) - E0s[h:].mean(0)
            rs = np.random.default_rng(abs(hash((p, t))) % (2**32))
            per = max(2, BUDGET // 2)
            # control mean per donor, excluding the gold control rows h:n2
            cmask = np.ones(len(S0[donors[0]]), dtype=bool); cmask[h:n2] = False
            c_dis = {m: S0[m][cmask].mean(0) for m in donors}
            # leaky arms: E34's exact design (full row range)
            b_inf_lk = np.mean([Sp[p][m].mean(0) - S0[m].mean(0)
                                for m in donors], axis=0)
            b_k_lk = np.mean([Sp[p][m][rs.permutation(len(Sp[p][m]))[:per]].mean(0)
                              - S0[m].mean(0) for m in donors], axis=0)
            # disjoint arms: donors see only rows [:h], never the gold cells
            b_inf_ds = np.mean([Sp[p][m][:h].mean(0) - c_dis[m]
                                for m in donors], axis=0)
            b_k_ds = np.mean([Sp[p][m][rs.permutation(h)[:min(per, h)]].mean(0)
                              - c_dis[m] for m in donors], axis=0)
            sn = rs.permutation(h)[:min(BUDGET, h)]
            nat = E1s[:h][sn].mean(0) - E0s[:h].mean(0)
            ceil = qlib.cos(E1s[:h].mean(0) - E0s[:h].mean(0), tgt)
            rec.append({"p": p, "t": t, "tgt": tgt, "nat": nat, "ceil": ceil,
                        "b_k_lk": b_k_lk, "bias_lk": b_inf_lk - tgt,
                        "b_k_ds": b_k_ds, "bias_ds": b_inf_ds - tgt})

    res = {k: [] for k in ["donor_lk", "deb_lk", "donor_ds", "deb_ds", "native",
                           "ceil"]}
    ret = {t: {"deb_ds": ([], []), "donor_ds": ([], [])} for t in M}
    for t in M:
        sub = [r for r in rec if r["t"] == t]
        if len(sub) < 20:
            continue
        names = sorted({r["p"] for r in sub})
        fold = {p: i % 5 for i, p in enumerate(rng.permutation(names))}
        for f in range(5):
            tr = [r for r in sub if fold[r["p"]] != f]
            te = [r for r in sub if fold[r["p"]] == f]
            if len(tr) < 10 or not te:
                continue
            bh_lk = np.mean([r["bias_lk"] for r in tr], axis=0)
            bh_ds = np.mean([r["bias_ds"] for r in tr], axis=0)
            for r in te:
                res["donor_lk"].append(qlib.cos(r["b_k_lk"], r["tgt"]))
                res["deb_lk"].append(qlib.cos(r["b_k_lk"] - bh_lk, r["tgt"]))
                res["donor_ds"].append(qlib.cos(r["b_k_ds"], r["tgt"]))
                res["deb_ds"].append(qlib.cos(r["b_k_ds"] - bh_ds, r["tgt"]))
                res["native"].append(qlib.cos(r["nat"], r["tgt"]))
                res["ceil"].append(r["ceil"])
                ret[t]["deb_ds"][0].append(r["b_k_ds"] - bh_ds)
                ret[t]["deb_ds"][1].append(r["tgt"])
                ret[t]["donor_ds"][0].append(r["b_k_ds"])
                ret[t]["donor_ds"][1].append(r["tgt"])

    mean = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"\nRESULT {qlib.DATASET}")
    print(f"  ceiling(split-half, budget h) {mean['ceil']:+.3f}")
    print(f"  native                        {mean['native']:+.3f}")
    print(f"  donor    leaky {mean['donor_lk']:+.3f}   disjoint {mean['donor_ds']:+.3f}")
    print(f"  debiased leaky {mean['deb_lk']:+.3f}   disjoint {mean['deb_ds']:+.3f}")

    def ci(a, b):
        d = np.array(res[a]) - np.array(res[b])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"delta": float(d.mean()), "ci": [float(lo), float(hi)],
                "significant": bool(lo > 0)}

    leak_size = ci("deb_lk", "deb_ds")
    d1 = ci("deb_ds", "donor_ds")
    d2 = ci("deb_ds", "native")
    print(f"  leak size (leaky - disjoint, debiased) {leak_size['delta']:+.4f} "
          f"[{leak_size['ci'][0]:+.4f},{leak_size['ci'][1]:+.4f}]")
    print(f"  debiased - donor  (disjoint) {d1['delta']:+.4f} "
          f"[{d1['ci'][0]:+.4f},{d1['ci'][1]:+.4f}]{' SIG' if d1['significant'] else ''}")
    print(f"  debiased - native (disjoint) {d2['delta']:+.4f} "
          f"[{d2['ci'][0]:+.4f},{d2['ci'][1]:+.4f}]{' SIG' if d2['significant'] else ''}")

    out = {"dataset": qlib.DATASET, "means": mean, "leak_size": leak_size,
           "debiased_minus_donor_disjoint": d1,
           "debiased_minus_native_disjoint": d2, "retrieval": {}}
    for t in M:
        pr, tg = ret[t]["deb_ds"]
        if len(tg) < 15:
            continue
        gen_p = np.mean(pr, axis=0); gen_t = np.mean(tg, axis=0)
        a1, m1, n = retrieval(ret[t]["donor_ds"][0], ret[t]["donor_ds"][1])
        a2, m2, _ = retrieval(pr, tg)
        a3, m3, _ = retrieval([v - gen_p for v in pr], [v - gen_t for v in tg])
        out["retrieval"][t] = {"donor": {"top1": a1, "mean_rank": m1},
                               "debiased": {"top1": a2, "mean_rank": m2},
                               "debiased_centred": {"top1": a3, "mean_rank": m3},
                               "n": n, "chance": 1.0 / n}
        print(f"  retrieval[{t}] top-1: donor {a1:.3f}  debiased {a2:.3f}  "
              f"centred {a3:.3f}  (chance {1.0/n:.3f}, n={n})")
    qlib.save_result(f"e37_debias_leakfix_{qlib.DATASET}", out)
    print("E37_DONE", flush=True)


if __name__ == "__main__":
    main()
