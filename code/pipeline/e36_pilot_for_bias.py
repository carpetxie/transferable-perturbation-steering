"""E36: how large a pilot does the bias correction need?

Finding 37 showed that subtracting a cross-fitted mean bias rescues transfer on
all three datasets, beating native estimation everywhere. It requires target
labels for the perturbations used to estimate the bias, so the practical
question is how many.

Finding 24 asked the same of K-selection and found ten perturbations, with fewer
than ten being actively worse than not tuning. That answer was uncomfortable
because a pilot of three misled. This asks whether the bias estimate is better
behaved.

The bias appears to be close to a fixed offset (leave-one-out consistency 0.75
to 0.94), and an offset is a much easier thing to estimate than an optimum on a
curve: averaging a handful of noisy vectors converges quickly, whereas locating
a minimum from a few noisy points can point the wrong way entirely. So the
prediction is that this needs FEWER pilot perturbations than K-selection did,
and that a small pilot should not be harmful.

For each pilot size n: draw n perturbations, estimate the bias from those only,
apply to all remaining perturbations, and compare against uncorrected donor
transfer and against native estimation. Repeated over many draws so the answer
reflects pilot luck rather than one fortunate sample.
"""
import numpy as np, qlib

M = ["scgpt", "gf1", "gf2"]
K = 24
BUDGET = 24
PILOTS = [1, 2, 3, 5, 10, 20]
N_DRAW = 30


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

    rows = {t: {} for t in M}
    for p in perts:
        for t in M:
            donors = [m for m in M if m != t]
            E0, E1 = S0[t], Sp[p][t]
            n2 = min(len(E0), len(E1)); h = n2 // 2
            if h < 10:
                continue
            E0s, E1s = E0[:n2], E1[:n2]
            tgt = E1s[h:].mean(0) - E0s[h:].mean(0)
            b_inf = np.mean([Sp[p][m].mean(0) - S0[m].mean(0) for m in donors], axis=0)
            rs = np.random.default_rng(abs(hash((p, t))) % (2**32))
            per = max(2, BUDGET // 2)
            b_k = np.mean([Sp[p][m][rs.permutation(len(Sp[p][m]))[:per]].mean(0)
                           - S0[m].mean(0) for m in donors], axis=0)
            sn = rs.permutation(h)[:min(BUDGET, h)]
            rows[t][p] = {"tgt": tgt, "b_k": b_k, "bias": b_inf - tgt,
                          "nat": qlib.cos(E1s[:h][sn].mean(0) - E0s[:h].mean(0), tgt)}

    base_don, base_nat = [], []
    for t in M:
        for p, r in rows[t].items():
            base_don.append(qlib.cos(r["b_k"], r["tgt"]))
            base_nat.append(r["nat"])
    don_m, nat_m = float(np.mean(base_don)), float(np.mean(base_nat))

    print(f"\n{qlib.DATASET}: donor {don_m:.3f}   native {nat_m:.3f}")
    print(f"{'pilot n':>8} {'debiased':>10} {'vs donor':>10} {'vs native':>11} "
          f"{'draws>native':>13}")
    out = {"dataset": qlib.DATASET, "donor": don_m, "native": nat_m, "pilots": {}}
    for n in PILOTS:
        vals, winsn = [], []
        for _ in range(N_DRAW):
            per_draw = []
            for t in M:
                names = list(rows[t])
                if len(names) < n + 5:
                    continue
                pil = list(rng.choice(names, n, replace=False))
                bias_hat = np.mean([rows[t][p]["bias"] for p in pil], axis=0)
                for p in names:
                    if p in pil:
                        continue
                    r = rows[t][p]
                    per_draw.append(qlib.cos(r["b_k"] - bias_hat, r["tgt"]))
            if per_draw:
                vals.append(float(np.mean(per_draw)))
                winsn.append(float(np.mean(per_draw)) > nat_m)
        v = np.array(vals)
        out["pilots"][n] = {"debiased": float(v.mean()), "sd": float(v.std()),
                            "vs_donor": float(v.mean() - don_m),
                            "vs_native": float(v.mean() - nat_m),
                            "frac_draws_beat_native": float(np.mean(winsn))}
        print(f"{n:8d} {v.mean():10.3f} {v.mean()-don_m:+10.3f} "
              f"{v.mean()-nat_m:+11.3f} {np.mean(winsn):13.0%}")

    ok = [n for n in PILOTS if out["pilots"][n]["vs_native"] > 0
          and out["pilots"][n]["frac_draws_beat_native"] >= 0.9]
    out["min_pilot"] = min(ok) if ok else None
    out["verdict"] = (f"a pilot of {min(ok)} perturbations suffices to beat native"
                      if ok else "no tested pilot size reliably beats native")
    print(f"VERDICT: {out['verdict']}")
    qlib.save_result(f"e36_pilot_bias_{qlib.DATASET}", out)


if __name__ == "__main__":
    main()
