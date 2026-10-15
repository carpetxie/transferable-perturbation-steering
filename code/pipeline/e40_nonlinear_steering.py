"""E40: does a transplanted delta steer a NONLINEAR consumer?

Finding "2d" (E30/E31) showed a transplanted delta, pushed through a linear
decoder fitted on control cells, predicts measured differential expression.
The standing caveat: the decoder is linear, so dec(z0+d)-dec(z0) = dW and the
control state z0 cancels exactly. That is a linear readout of the delta, not a
steering test; it cannot fail no matter how wrong z0+d is as a STATE.

This closes that gap with the cheapest honest nonlinear consumer: an MLP
decoder (24 -> 256 -> 2000 HVG log-CPM, ReLU) fitted on CONTROL cells only,
mapping shared-space state to expression. For a nonlinear decoder the steered
state z0+d must lie somewhere the decoder treats sensibly, or
dec(z0+d)-dec(z0) is garbage. If the transplanted delta still predicts
measured DE through this consumer, the steered STATE (not just the delta) is
usable by a consumer that never saw a perturbed cell.

Cell discipline (rule 16, E37): all embeddings-side estimates (donor consensus,
native reference, z0) use the FIRST half of each condition's cells; measured
DE gold uses log-CPM from the SECOND half only. Control rows likewise split.

Arms, all scored as cos(dec-prediction, measured DE):
  transplant  d = donor-consensus delta (two other encoders, zero target labels)
  native      d = target's own first-half delta (reference, needs labels)
  mismatch    d for a DIFFERENT perturbation (specificity floor)
  generic     d = mean delta over all other perturbations
  linear      the transplant delta through a ridge (linear) decoder, same folds
              (the E30 baseline, for comparison on identical golds)
Also reported with the generic axis removed from prediction and gold, which is
the discriminative version.
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

    # ---- shared space on control cells
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

    # ---- expression side: control-HVG log-CPM per condition, split rows
    hvg, _, mu_x = qlib.fit_control_pca(idx, syms)

    def expr(cond):
        X = qlib.load_rows(idx[cond]).astype(np.float64)
        Xn = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
        return Xn[:, hvg] - mu_x

    Xc = expr(C)
    nC = min(len(Xc), len(S0[M[0]]))
    hC = nC // 2
    # decoder training set: control FIRST half (embeddings + expression)
    Xc_tr = Xc[:hC]
    x_gold_ctrl = Xc[hC:nC].mean(0)          # control expression, gold half

    # measured DE gold per pert: second-half expression minus control gold half
    de_gold, x_first = {}, {}
    for p in perts:
        Xp = expr(p)
        n2 = min(len(Xp), len(Sp[p][M[0]])); h = n2 // 2
        if h < 10:
            continue
        de_gold[p] = Xp[h:n2].mean(0) - x_gold_ctrl
        x_first[p] = h
    perts = [p for p in perts if p in de_gold]

    results = {}
    for t in M:
        donors = [m for m in M if m != t]
        Z_tr = S0[t][:hC]
        z0 = Z_tr.mean(0)

        # nonlinear decoder on control first half
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

        # linear decoder (ridge) on the same training rows
        W_lin, _ = qlib.ridge_map(Z_tr, Xc_tr, seed=0)

        def mlp_pred(d):
            with torch.no_grad():
                a = dec(torch.tensor((z0 + d)[None, :], dtype=torch.float32))
                b = dec(torch.tensor(z0[None, :], dtype=torch.float32))
            return (a - b).numpy()[0]

        # deltas per pert, first-half cells only
        d_don, d_nat = {}, {}
        for p in perts:
            h = x_first[p]
            d_don[p] = np.mean([Sp[p][m][:h].mean(0) - S0[m][:hC].mean(0)
                                for m in donors], axis=0)
            d_nat[p] = Sp[p][t][:h].mean(0) - S0[t][:hC].mean(0)
        d_gen = np.mean(list(d_don.values()), axis=0)

        arms = {k: [] for k in ["transplant", "native", "mismatch", "generic",
                                "linear"]}
        preds_t, golds = [], []
        pl = list(perts)
        mism = {p: pl[(i + 1) % len(pl)] for i, p in enumerate(pl)}
        for p in perts:
            gold = de_gold[p]
            pt = mlp_pred(d_don[p])
            arms["transplant"].append(qlib.cos(pt, gold))
            arms["native"].append(qlib.cos(mlp_pred(d_nat[p]), gold))
            arms["mismatch"].append(qlib.cos(mlp_pred(d_don[mism[p]]), gold))
            arms["generic"].append(qlib.cos(mlp_pred(d_gen), gold))
            arms["linear"].append(qlib.cos(d_don[p] @ W_lin, gold))
            preds_t.append(pt); golds.append(gold)

        # generic-removed versions for the nonlinear transplant arm
        gp = np.mean(preds_t, axis=0); gg = np.mean(golds, axis=0)
        cen_tr = [qlib.cos(a - gp, b - gg) for a, b in zip(preds_t, golds)]
        cen_mm = [qlib.cos(preds_t[pl.index(mism[p])] - gp, de_gold[p] - gg)
                  for p in pl]

        row = {k: float(np.mean(v)) for k, v in arms.items()}
        row["transplant_centred"] = float(np.mean(cen_tr))
        row["mismatch_centred"] = float(np.mean(cen_mm))
        d = np.array(cen_tr) - np.array(cen_mm)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        row["centred_delta_vs_mismatch"] = {
            "delta": float(d.mean()), "ci": [float(lo), float(hi)],
            "significant": bool(lo > 0)}
        results[t] = row
        print(f"\n[{qlib.DATASET} target={t}] nonlinear steering, "
              f"{len(perts)} perts")
        for k in ["transplant", "native", "mismatch", "generic", "linear"]:
            print(f"  {k:<11} {row[k]:+.3f}")
        print(f"  centred: transplant {row['transplant_centred']:+.3f}  "
              f"mismatch {row['mismatch_centred']:+.3f}  "
              f"delta {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"{' SIG' if lo > 0 else ''}", flush=True)

    ok = [t for t in results
          if results[t]["centred_delta_vs_mismatch"]["significant"]]
    out = {"dataset": qlib.DATASET, "targets": results,
           "verdict": (f"steered state usable by a nonlinear consumer for "
                       f"targets {ok}" if len(ok) == len(results) else
                       f"partial: significant only for {ok}" if ok else
                       "nonlinear consumer rejects the steered state")}
    print(f"\nVERDICT: {out['verdict']}")
    qlib.save_result(f"e40_nonlinear_steering_{qlib.DATASET}", out)
    print("E40_DONE", flush=True)


if __name__ == "__main__":
    main()
