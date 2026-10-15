"""E41: does the debiased transplant survive all the way to gene space?

The two newest positives compose naturally and nothing has tested the
composition. Finding 37/E37: subtracting a cross-fitted bias (one-pilot
regime) makes shared-space transfer beat native on all three datasets,
leak-free. Finding 40/E40: a transplanted delta steers a NONLINEAR
control-trained decoder to correct differential expression on Norman.

Open question: on Papalexi, where the RAW transplant is anti-correlated
(donor cosine -0.24 disjoint), does the debiased delta produce correct
gene-level DE through the nonlinear consumer? If yes, the deployment story is
complete end-to-end: donor encoders + one labelled pilot perturbation in the
target + control cells = gene-level DE predictions for unlabelled
perturbations, in a new cell type, through a consumer that never saw one.

Protocol: E40's decoder and cell discipline (states from first-half cells, DE
gold from second-half cells, MLP fitted on control first half), E37's
disjoint-cell donor estimates, E34's pert-level 5-fold cross-fitted bias
(shared-space bias estimated against first-half-cell native deltas of TRAINING
perturbations only, so the pilot never sees a test perturbation and never sees
gold cells).

Arms (cos to measured DE): raw transplant, debiased transplant, native
reference, debiased-mismatch (wrong perturbation, debiased with the same
bias_hat -- the specificity floor), generic. Centred versions for the
discriminative comparison.
"""
import numpy as np, torch, qlib

M = ["scgpt", "gf1", "gf2"]
K = 24
MIN_CELLS = 60
SEED = 0


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
    perts = [c for c in sorted(idx.keys() - {C}) if len(idx[c]) >= MIN_CELLS]
    Sp = {p: {m: S(p, m) for m in M} for p in perts}

    hvg, _, mu_x = qlib.fit_control_pca(idx, syms)

    def expr(cond):
        X = qlib.load_rows(idx[cond]).astype(np.float64)
        Xn = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
        return Xn[:, hvg] - mu_x

    Xc = expr(C)
    nC = min(len(Xc), len(S0[M[0]])); hC = nC // 2
    Xc_tr = Xc[:hC]
    x_gold_ctrl = Xc[hC:nC].mean(0)

    de_gold, halves = {}, {}
    for p in perts:
        Xp = expr(p)
        n2 = min(len(Xp), len(Sp[p][M[0]])); h = n2 // 2
        if h < 10:
            continue
        de_gold[p] = Xp[h:n2].mean(0) - x_gold_ctrl
        halves[p] = h
    perts = [p for p in perts if p in de_gold]

    results = {}
    for t in M:
        donors = [m for m in M if m != t]
        Z_tr = S0[t][:hC]; z0 = Z_tr.mean(0)

        torch.manual_seed(SEED)
        dec = torch.nn.Sequential(
            torch.nn.Linear(K, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, len(hvg)))
        opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
        Zt = torch.tensor(Z_tr, dtype=torch.float32)
        Xt = torch.tensor(Xc_tr, dtype=torch.float32)
        for ep in range(200):
            perm = torch.randperm(len(Zt))
            for i in range(0, len(Zt), 256):
                bix = perm[i:i + 256]
                loss = torch.nn.functional.mse_loss(dec(Zt[bix]), Xt[bix])
                opt.zero_grad(); loss.backward(); opt.step()
        dec.eval()

        def mlp_pred(d):
            with torch.no_grad():
                a = dec(torch.tensor((z0 + d)[None, :], dtype=torch.float32))
                b = dec(torch.tensor(z0[None, :], dtype=torch.float32))
            return (a - b).numpy()[0]

        # disjoint-cell shared-space deltas + per-pert bias vs native delta
        d_don, d_nat = {}, {}
        for p in perts:
            h = halves[p]
            d_don[p] = np.mean([Sp[p][m][:h].mean(0) - S0[m][:hC].mean(0)
                                for m in donors], axis=0)
            d_nat[p] = Sp[p][t][:h].mean(0) - S0[t][:hC].mean(0)
        bias = {p: d_don[p] - d_nat[p] for p in perts}

        # pert-level 5-fold cross-fitted debiasing
        fold = {p: i % 5 for i, p in enumerate(rng.permutation(sorted(perts)))}
        d_deb = {}
        for f in range(5):
            tr = [p for p in perts if fold[p] != f]
            te = [p for p in perts if fold[p] == f]
            if len(tr) < 10:
                continue
            bh = np.mean([bias[p] for p in tr], axis=0)
            for p in te:
                d_deb[p] = d_don[p] - bh
        ps = [p for p in perts if p in d_deb]
        d_gen = np.mean([d_don[p] for p in ps], axis=0)
        mism = {p: ps[(i + 1) % len(ps)] for i, p in enumerate(ps)}

        arms = {k: [] for k in ["raw", "debiased", "native", "mismatch_deb",
                                "generic"]}
        preds, golds = [], []
        for p in ps:
            gold = de_gold[p]
            pd = mlp_pred(d_deb[p])
            arms["raw"].append(qlib.cos(mlp_pred(d_don[p]), gold))
            arms["debiased"].append(qlib.cos(pd, gold))
            arms["native"].append(qlib.cos(mlp_pred(d_nat[p]), gold))
            arms["mismatch_deb"].append(qlib.cos(mlp_pred(d_deb[mism[p]]), gold))
            arms["generic"].append(qlib.cos(mlp_pred(d_gen), gold))
            preds.append(pd); golds.append(gold)

        gp = np.mean(preds, axis=0); gg = np.mean(golds, axis=0)
        cen_de = [qlib.cos(a - gp, b - gg) for a, b in zip(preds, golds)]
        cen_mm = [qlib.cos(preds[ps.index(mism[p])] - gp, de_gold[p] - gg)
                  for p in ps]
        d = np.array(cen_de) - np.array(cen_mm)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])

        row = {k: float(np.mean(v)) for k, v in arms.items()}
        row["debiased_centred"] = float(np.mean(cen_de))
        row["mismatch_centred"] = float(np.mean(cen_mm))
        row["centred_delta"] = {"delta": float(d.mean()),
                                "ci": [float(lo), float(hi)],
                                "significant": bool(lo > 0)}
        results[t] = row
        print(f"\n[{qlib.DATASET} target={t}] {len(ps)} perts")
        for k in ["raw", "debiased", "native", "mismatch_deb", "generic"]:
            print(f"  {k:<13} {row[k]:+.3f}")
        print(f"  centred: debiased {row['debiased_centred']:+.3f}  "
              f"mismatch {row['mismatch_centred']:+.3f}  "
              f"delta {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"{' SIG' if lo > 0 else ''}", flush=True)

    ok = [t for t in results if results[t]["centred_delta"]["significant"]]
    out = {"dataset": qlib.DATASET, "targets": results,
           "verdict": (f"debiased transplant yields specific gene-level DE for "
                       f"{ok}" if ok else "no specific gene-level signal")}
    print(f"\nVERDICT [{qlib.DATASET}]: {out['verdict']}")
    qlib.save_result(f"e41_debias_steering_{qlib.DATASET}", out)
    print("E41_DONE", flush=True)


if __name__ == "__main__":
    main()
