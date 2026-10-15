"""E35: is the debiasing gain perturbation-specific, or just the generic direction?

E34 showed that subtracting a cross-fitted mean bias rescues transfer on all
three datasets, including Papalexi where it had reversed. The gain is large
enough to be suspicious: debiased cosine reaches 0.885 on Papalexi against a
noise ceiling of 0.579.

There is a specific mechanism that would inflate it. The bias estimate is
mean(b_inf(p) - tgt(p)) over training perturbations. If target effects share a
common direction, that mean contains minus-the-generic-direction, and
subtracting it adds the generic direction back into every prediction. Held-out
targets contain that same generic component, so cosine rises without any
perturbation-specific information being recovered.

Cosine to the correct target cannot distinguish these. Retrieval can. Rank ALL
perturbations by agreement with a prediction and ask whether the correct one
comes first. A generic-only signal scores high cosine and chance retrieval.

Reported for donor, debiased, and debiased after removing the generic direction
from both prediction and target, which is the strictest version.
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
    Sp = {p: {m: S(p, m) for m in M} for p in perts}

    rows = {t: [] for t in M}
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
            rows[t].append({"p": p, "tgt": tgt, "b_k": b_k, "bias": b_inf - tgt})

    def retrieval(preds, targs):
        """top-1 and mean rank of the correct target among all targets."""
        n = len(preds)
        P = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in preds])
        T = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in targs])
        Sim = P @ T.T
        order = np.argsort(-Sim, axis=1)
        rk = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
        return float((rk == 1).mean()), float(rk.mean()), n

    agg = {}
    for t in M:
        sub = rows[t]
        if len(sub) < 25:
            continue
        names = sorted({r["p"] for r in sub})
        fold = {p: i % 5 for i, p in enumerate(rng.permutation(names))}
        dn, db, tg = [], [], []
        for f in range(5):
            tr = [r for r in sub if fold[r["p"]] != f]
            te = [r for r in sub if fold[r["p"]] == f]
            if len(tr) < 10 or not te:
                continue
            bias_hat = np.mean([r["bias"] for r in tr], axis=0)
            for r in te:
                dn.append(r["b_k"]); db.append(r["b_k"] - bias_hat); tg.append(r["tgt"])
        if len(tg) < 15:
            continue
        gen_p = np.mean(db, axis=0); gen_t = np.mean(tg, axis=0)
        agg[t] = {
            "donor": retrieval(dn, tg),
            "debiased": retrieval(db, tg),
            "debiased_centred": retrieval([v - gen_p for v in db],
                                          [v - gen_t for v in tg]),
        }

    print(f"\n{qlib.DATASET}: retrieval of the correct perturbation "
          f"(chance top-1 = 1/n)")
    print(f"{'target':<8} {'arm':<18} {'top-1':>7} {'mean rank':>10} {'n':>5}")
    out = {"dataset": qlib.DATASET, "targets": {}}
    for t, d in agg.items():
        out["targets"][t] = {}
        for arm in ("donor", "debiased", "debiased_centred"):
            a, mr, n = d[arm]
            out["targets"][t][arm] = {"top1": a, "mean_rank": mr, "n": n,
                                      "chance": 1.0 / n}
            print(f"{t:<8} {arm:<18} {a:7.3f} {mr:10.2f} {n:5d}")
    # pooled
    for arm in ("donor", "debiased", "debiased_centred"):
        v = [out["targets"][t][arm]["top1"] for t in out["targets"]]
        ch = [out["targets"][t][arm]["chance"] for t in out["targets"]]
        print(f"POOLED {qlib.DATASET:9s} {arm:<18} top-1 {np.mean(v):.3f} "
              f"(chance {np.mean(ch):.3f})")
    qlib.save_result(f"e35_debias_specificity_{qlib.DATASET}", out)


if __name__ == "__main__":
    main()
