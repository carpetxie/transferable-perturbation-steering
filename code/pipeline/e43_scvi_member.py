"""E43: scVI (a real VAE) and a domain-shifted checkpoint join the space.

E38/E39 established on the local pipeline that expression PCA and even a
random-init autoencoder join the shared space at ~80% of their own ceilings,
with a modest premium held only by pretrained encoders. Two views were still
missing, and the Modal run (validated functionally by e34_verify_modal, all
quantities within 0.02 of local) produced both:

  scvi           an actual scVI model (VAE, NB likelihood, dim 64), trained
                 from scratch on this dataset's control cells on GPU. The
                 HANDOFF's original "add scVI" item, done properly.
  geneformer_cl  Geneformer-V2-104M_CLcancer: same architecture and vocab as
                 gf2 but continually pretrained on cancer data. A
                 domain-shifted family member, never used before.

Everything here uses the MODAL embedding set for every view, so there is no
pipeline mixing. Protocol is E38's: seven-view GCCA (K=24) on control cells,
gold effects from rows [h:], donor estimates from rows [:h] only, shuffled
floor per target. FM donor pool for each target = the four original FMs
minus the target (scvi, pca and geneformer_cl are never donors here, so
their rows measure pure transfer INTO them from the standard pool).
"""
import os
import pathlib
import numpy as np
import qlib

MODAL_EMB = pathlib.Path(os.environ.get(
    "MODAL_EMB", pathlib.Path.home() / ".cache/silico_vec2vec/modal_emb/norman"))

FM = ["scgpt", "geneformer", "geneformer_v2", "geneformer_v3"]
EXTRA = ["geneformer_cl", "scvi", "pca"]
M = FM + EXTRA
K = 24
MIN_CELLS = 60


def load_modal(cond, model):
    f = MODAL_EMB / f"{cond.replace('/','-')}__{model}.npz"
    d = np.load(f)
    return d["Z"].astype(np.float64), d["lib"], d["ngene"]


def main():
    idx, syms, _ = qlib.condition_index()
    rng = np.random.default_rng(0)
    C = qlib.CONTROL

    betas, mus, views = {}, {}, {}
    for m in M:
        Z, lib, ng = load_modal(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)
    S = lambda c, m: ((qlib.apply_depth_projector(*load_modal(c, m), betas[m])
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
    print(f"norman/modal: {len(perts)} perturbations ({miss} skipped), "
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

    out = {"dataset": "norman_modal", "K": K, "targets": {}}
    per = {}
    common = sorted(set.intersection(*[set(rec[t]) for t in M]))
    print(f"\n{'target':<15} {'ceiling':>8} {'FM-donors':>10} {'%ceil':>7} "
          f"{'shuffled':>9}")
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
        out["targets"][t] = {"n_perts": len(ps), "ceiling": mc,
                             "fm_donors": mt, "fm_pct_ceiling": mt / mc,
                             "shuffled_floor": mf}
        print(f"{t:<15} {mc:8.3f} {mt:10.3f} {mt/mc:7.1%} {mf:9.3f}")

    # paired bootstrap of each row's %-of-ceiling vs pca's (the linear floor)
    if "pca" in per:
        cr, trr = per["pca"]
        n = len(cr)
        out["premium_vs_pca"] = {}
        print("\npremium over PCA (paired bootstrap on perts)")
        for t in M:
            if t == "pca" or t not in per:
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
            out["premium_vs_pca"][t] = {
                "delta_pct": d, "ci": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0)}
            print(f"  {t:<15} {d:+7.1%} [{lo:+7.1%},{hi:+7.1%}]"
                  f"{' SIG' if lo > 0 or hi < 0 else ''}")

    sv = out["targets"].get("scvi", {}).get("fm_pct_ceiling")
    cl = out["targets"].get("geneformer_cl", {}).get("fm_pct_ceiling")
    pct = [out["targets"][t]["fm_pct_ceiling"] for t in FM
           if t in out["targets"]]
    out["verdict"] = (f"scVI {sv:.0%}, CLcancer {cl:.0%} of own ceiling "
                      f"(FM band {min(pct):.0%}-{max(pct):.0%})")
    print(f"\nVERDICT: {out['verdict']}")
    qlib.save_result("e43_scvi_member_norman", out)
    print("E43_DONE", flush=True)


if __name__ == "__main__":
    main()
