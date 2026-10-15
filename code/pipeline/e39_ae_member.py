"""E39: does a NONLINEAR, non-pretrained encoder also join the shared space?

Finding 39 (E38) showed control-fitted expression PCA is a full member of the
shared space: FM-measured effects transfer into it at the same fraction of its
own ceiling as into a cross-family transformer. PCA is linear, so one reading
survives: maybe the invariance is specifically about LINEAR views of log-CPM,
and any nonlinear representation learned from scratch would join badly.

HANDOFF next-step 3 names scVI; installing scvi-tools is ~1 GB of dependencies
on a disk with 5.5 GB free, and the conceptual content of that test is carried
by any nonlinear encoder trained from scratch on these cells. So: a small
autoencoder (2000 HVG -> 512 -> 128 -> 512 -> 2000, ReLU, MSE on centred
log-CPM), trained on CONTROL cells only, fixed seed. No pretraining corpus, no
tokenisation, nonlinear. Its 128-d latent becomes a sixth view.

The matched-dimension control this needs (LESSONS rule: include a control that
separates "the act of projecting" from "the learned geometry"): the SAME
architecture at random initialisation, never trained. If the random encoder
joins the space as well as the trained one, membership is trivial and says
nothing; if trained ~ FM band and random is clearly worse, joining reflects
learned faithfulness to the cells.

Protocol otherwise identical to E38: six-view GCCA (K=24) on control cells,
gold = rows [h:], donors = rows [:h] (E37 discipline), shuffled floor.
"""
import numpy as np, torch, qlib

FM = ["scgpt", "gf1", "gf2", "gf3"]
K = 24
MIN_CELLS = 60
SEED = 0


def logcpm_hvg(cond, idx, syms):
    hvg, _, mu = qlib.fit_control_pca(idx, syms)
    X = qlib.load_rows(idx[cond]).astype(np.float64)
    lib = X.sum(1)
    ngene = (X > 0).sum(1).astype(np.float32)
    Xn = np.log1p(X / np.maximum(lib[:, None], 1) * 1e4)
    return (Xn[:, hvg] - mu), lib, ngene


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = torch.nn.Sequential(
            torch.nn.Linear(2000, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 128))
        self.d = torch.nn.Sequential(
            torch.nn.Linear(128, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 2000))

    def forward(self, x):
        z = self.e(x)
        return z, self.d(z)


def train_ae(Xc):
    torch.manual_seed(SEED)
    ae = Enc()
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    X = torch.tensor(Xc, dtype=torch.float32)
    n = len(X)
    for ep in range(60):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            b = X[perm[i:i + 256]]
            _, rec = ae(b)
            loss = torch.nn.functional.mse_loss(rec, b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 20 == 0:
            print(f"  ae epoch {ep}: mse {tot/n:.4f}", flush=True)
    ae.eval()
    return ae


def main():
    idx, syms, _ = qlib.condition_index()
    rng = np.random.default_rng(0)
    C = qlib.CONTROL

    Xc, _, _ = logcpm_hvg(C, idx, syms)
    ae = train_ae(Xc)
    torch.manual_seed(SEED + 1)
    ae_rand = Enc(); ae_rand.eval()

    def embed(model, cond):
        Xn, lib, ng = logcpm_hvg(cond, idx, syms)
        with torch.no_grad():
            z, _ = model(torch.tensor(Xn, dtype=torch.float32))
        return z.numpy().astype(np.float64), lib, ng

    M = FM + ["pca", "ae", "ae_rand"]

    def raw(cond, m):
        if m == "pca":
            return qlib.pca_embed_condition(cond, idx, syms)
        if m == "ae":
            return embed(ae, cond)
        if m == "ae_rand":
            return embed(ae_rand, cond)
        return qlib.load_emb(cond, m)

    betas, mus, views = {}, {}, {}
    for m in M:
        Z, lib, ng = raw(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)
    S = lambda c, m: ((qlib.apply_depth_projector(*raw(c, m), betas[m])
                       - mus[m]) @ g["P"][m]) @ g["W"][m] @ g["C"][m]
    S0 = {m: S(C, m) for m in M}
    perts = [c for c in sorted(idx.keys() - {C}) if len(idx[c]) >= MIN_CELLS]
    Sp, miss = {}, 0
    for p in perts:
        try:
            Sp[p] = {m: S(p, m) for m in M}
        except FileNotFoundError:
            miss += 1
    perts = [p for p in perts if p in Sp]
    print(f"{qlib.DATASET}: {len(perts)} perturbations ({miss} skipped), "
          f"views {M}", flush=True)

    rec = {t: {} for t in M}
    for p in perts:
        for t in M:
            E0, E1 = S0[t], Sp[p][t]
            n2 = min(len(E0), len(E1)); h = n2 // 2
            if h < 10:
                continue
            tgt = E1[h:n2].mean(0) - E0[h:n2].mean(0)
            cmask = np.ones(len(E0), dtype=bool); cmask[h:n2] = False
            dons = {m: Sp[p][m][:h].mean(0) - S0[m][cmask].mean(0)
                    for m in M if m != t}
            rec[t][p] = {"tgt": tgt, "dons": dons,
                         "self": E1[:h].mean(0) - E0[:h].mean(0)}

    out = {"dataset": qlib.DATASET, "K": K, "targets": {}}
    per = {}  # per-target per-pert (ceiling, transfer) arrays, aligned on perts
    print(f"\n{'target':<8} {'ceiling':>8} {'FM-donors':>10} {'%ceil':>7} "
          f"{'shuffled':>9}")
    common = sorted(set.intersection(*[set(rec[t]) for t in M]))
    for t in M:
        sub = rec[t]; ps = common
        if len(ps) < 20:
            continue
        fm_d = [m for m in FM if m != t]
        ceil = np.array([qlib.cos(sub[p]["self"], sub[p]["tgt"]) for p in ps])
        tr = np.array([qlib.cos(np.mean([sub[p]["dons"][m] for m in fm_d],
                                        axis=0), sub[p]["tgt"]) for p in ps])
        sh = rng.permutation(len(ps))
        sh = np.where(sh == np.arange(len(ps)), (sh + 1) % len(ps), sh)
        fl = [qlib.cos(np.mean([sub[ps[sh[i]]]["dons"][m] for m in fm_d],
                               axis=0), sub[ps[i]]["tgt"])
              for i in range(len(ps))]
        per[t] = (ceil, tr)
        mc, mt, mf = float(ceil.mean()), float(tr.mean()), float(np.mean(fl))
        out["targets"][t] = {"n_perts": len(ps), "ceiling": mc, "fm_donors": mt,
                             "fm_pct_ceiling": mt / mc, "shuffled_floor": mf}
        print(f"{t:<8} {mc:8.3f} {mt:10.3f} {mt/mc:7.1%} {mf:9.3f}")

    # paired bootstrap over perturbations: each target's %-of-ceiling minus the
    # random-init encoder's, resampling the SAME perturbations for both
    if "ae_rand" in per:
        cr, trr = per["ae_rand"]
        n = len(cr)
        out["premium_vs_random"] = {}
        print("\npremium over random-init encoder (paired bootstrap on perts)")
        for t in M:
            if t == "ae_rand" or t not in per:
                continue
            ct, tt = per[t]
            ds = []
            for _ in range(4000):
                ix = rng.integers(0, n, n)
                ds.append(tt[ix].mean() / ct[ix].mean()
                          - trr[ix].mean() / cr[ix].mean())
            ds = np.array(ds)
            lo, hi = np.percentile(ds, [2.5, 97.5])
            d = float(tt.mean() / ct.mean() - trr.mean() / cr.mean())
            out["premium_vs_random"][t] = {
                "delta_pct": d, "ci": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0)}
            print(f"  {t:<8} {d:+7.1%} [{lo:+7.1%},{hi:+7.1%}]"
                  f"{' SIG' if lo > 0 or hi < 0 else ''}")

    ta = out["targets"].get("ae", {}).get("fm_pct_ceiling")
    tr_ = out["targets"].get("ae_rand", {}).get("fm_pct_ceiling")
    pct = [out["targets"][t]["fm_pct_ceiling"] for t in FM if t in out["targets"]]
    out["verdict"] = (f"trained AE {ta:.0%} of ceiling vs random-init {tr_:.0%} "
                      f"(FM band {min(pct):.0%}-{max(pct):.0%})")
    print(f"VERDICT: {out['verdict']}")
    qlib.save_result(f"e39_ae_member_{qlib.DATASET}", out)
    print("E39_DONE", flush=True)


if __name__ == "__main__":
    main()
