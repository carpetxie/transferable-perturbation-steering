"""E34: if the shared error is systematic, can it be estimated and removed?

Finding 36 established that the encoders' sampling errors are uncorrelated
(split-half rho = 0.00) while their total error correlation is 0.62 to 0.85. The
shared component is therefore systematic bias, not noise. Systematic errors have
a property noise does not: they can be consistent, and anything consistent can
be estimated from examples and subtracted.

So: is the donor bias consistent ACROSS perturbations? If a single shared bias
direction explains a meaningful share of it, then estimating that direction on
perturbations where target labels exist and subtracting it from predictions for
perturbations where they do not should improve transfer.

MEASURED FIRST, then applied:

  (a) consistency  -- how much of each perturbation's bias is explained by the
                      mean bias across all others. Leave-one-out, so a
                      perturbation never contributes to the bias estimate used
                      to correct it.
  (b) correction   -- cross-fitted. Estimate the mean bias on TRAINING
                      perturbations, subtract from held-out donor predictions,
                      and compare against uncorrected donors and against native.

This is the honest version of the obvious idea. The bias estimate needs target
labels for the training perturbations, so this is no longer a zero-label method;
it is the labelled-pilot regime that Finding 34 concluded was necessary anyway.
The question is whether a pilot buys back the transfer advantage.

Run on all three datasets. Papalexi is the decisive one: bias there is near
total (bias fraction 0.85), so if debiasing rescues transfer anywhere it should
show there most clearly.
"""
import numpy as np, qlib

M = ["scgpt", "gf1", "gf2"]
K = 24
BUDGET = 24


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

    # per (perturbation, target): donor limit, target effect, bias
    rec = []
    for p in perts:
        Sp = {m: S(p, m) for m in M}
        for t in M:
            donors = [m for m in M if m != t]
            E0, E1 = S0[t], Sp[t]
            n2 = min(len(E0), len(E1)); h = n2 // 2
            if h < 10:
                continue
            E0s, E1s = E0[:n2], E1[:n2]
            tgt = E1s[h:].mean(0) - E0s[h:].mean(0)
            b_inf = np.mean([Sp[m].mean(0) - S0[m].mean(0) for m in donors], axis=0)
            rs = np.random.default_rng(abs(hash((p, t))) % (2**32))
            per = max(2, BUDGET // 2)
            b_k = np.mean([Sp[m][rs.permutation(len(Sp[m]))[:per]].mean(0)
                           - S0[m].mean(0) for m in donors], axis=0)
            sn = rs.permutation(h)[:min(BUDGET, h)]
            nat = E1s[:h][sn].mean(0) - E0s[:h].mean(0)
            rec.append({"p": p, "t": t, "tgt": tgt, "b_inf": b_inf,
                        "b_k": b_k, "nat": nat, "bias": b_inf - tgt})

    # ---- (a) consistency of the bias, leave-one-out within each target
    cons = []
    for t in M:
        sub = [r for r in rec if r["t"] == t]
        if len(sub) < 10:
            continue
        B = np.stack([r["bias"] for r in sub])
        for i in range(len(sub)):
            others = np.delete(B, i, axis=0).mean(0)
            cons.append(qlib.cos(others, B[i]))
    consistency = float(np.mean(cons)) if cons else float("nan")

    # ---- (b) cross-fitted debiasing
    res = {"donor": [], "debiased": [], "native": []}
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
            bias_hat = np.mean([r["bias"] for r in tr], axis=0)
            for r in te:
                res["donor"].append(qlib.cos(r["b_k"], r["tgt"]))
                res["debiased"].append(qlib.cos(r["b_k"] - bias_hat, r["tgt"]))
                res["native"].append(qlib.cos(r["nat"], r["tgt"]))

    print(f"RESULT {qlib.DATASET:9s} bias-consistency {consistency:+.3f}  "
          f"donor {np.mean(res['donor']):+.3f}  debiased {np.mean(res['debiased']):+.3f}  "
          f"native {np.mean(res['native']):+.3f}", flush=True)
    d = np.array(res["debiased"]) - np.array(res["donor"])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    d2 = np.array(res["debiased"]) - np.array(res["native"])
    bs2 = [d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(4000)]
    lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
    print(f"       debiased - donor  {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
          f"{' SIG' if lo > 0 else ''}")
    print(f"       debiased - native {d2.mean():+.4f} [{lo2:+.4f},{hi2:+.4f}]"
          f"{' SIG' if lo2 > 0 else ''}")
    qlib.save_result(f"e34_debias_{qlib.DATASET}", {
        "dataset": qlib.DATASET, "bias_consistency": consistency,
        "donor": float(np.mean(res["donor"])),
        "debiased": float(np.mean(res["debiased"])),
        "native": float(np.mean(res["native"])),
        "debiased_minus_donor": {"delta": float(d.mean()), "ci": [float(lo), float(hi)],
                                 "significant": bool(lo > 0)},
        "debiased_minus_native": {"delta": float(d2.mean()), "ci": [float(lo2), float(hi2)],
                                  "significant": bool(lo2 > 0)}})


if __name__ == "__main__":
    main()
