"""E38: is coordinate invariance about foundation models, or about cell
representations in general?

Every positive transfer result so far lives in a GCCA space fitted over
transformer encoders (one scGPT + three Geneformers). HANDOFF section 7 item 3
names the strongest cheap test of generality: promote control-fitted expression
PCA from a baseline to a FULL MEMBER of the shared space. PCA shares nothing
with the encoders except the counts matrix: no pretraining corpus, no
tokenisation, no attention, not even nonlinearity. If a linear projection of
log-CPM joins the shared space and

  (a) perturbation effects transfer INTO it from the encoders at a comparable
      fraction of its own noise ceiling, and
  (b) it works as a donor for the encoders without hurting them,

then the invariance is a property of faithful representations of these cells,
not of transformers. If instead PCA transfers poorly relative to its ceiling,
the encoders share something PCA lacks and the transformer framing stands.

Design is E33's fixed-space protocol with E37's cell discipline:
  - one GCCA (K=24) fitted on CONTROL cells of all five views, held fixed
  - gold effect for a target = rows [h:] of its cells
  - donor estimates use rows [:h] only, never the gold cells (rule 16)
  - native ceiling per row = the target's own rows [:h] estimate vs gold
  - floors: shuffled perturbation labels (generic floor) per target

Norman only: PCA needs the raw counts, which exist locally for Norman alone.
"""
import numpy as np, qlib

FM = ["scgpt", "gf1", "gf2", "gf3"]
M = FM + ["pca"]
K = 24
MIN_CELLS = 60


def main():
    idx, syms, _ = qlib.condition_index()
    rng = np.random.default_rng(0)
    C = qlib.CONTROL

    def raw(cond, m):
        if m == "pca":
            return qlib.pca_embed_condition(cond, idx, syms)
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
    Sp, missing = {}, 0
    for p in perts:
        try:
            Sp[p] = {m: S(p, m) for m in M}
        except FileNotFoundError:
            missing += 1
    perts = [p for p in perts if p in Sp]
    print(f"{qlib.DATASET}: {len(perts)} perturbations "
          f"({missing} skipped, missing a view), views {M}", flush=True)

    # per (pert, target): gold, per-donor disjoint estimates, self estimate
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
            self_est = E1[:h].mean(0) - E0[:h][: max(h, 1)].mean(0)
            rec[t][p] = {"tgt": tgt, "dons": dons, "self": self_est}

    out = {"dataset": qlib.DATASET, "K": K, "targets": {}}
    print(f"\n{'target':<7} {'ceiling':>8} {'FM-donors':>10} {'%ceil':>6} "
          f"{'all-donors':>11} {'%ceil':>6} {'shuffled':>9}")
    for t in M:
        sub = rec[t]
        ps = sorted(sub)
        if len(ps) < 20:
            continue
        fm_d = [m for m in FM if m != t]
        ceil = [qlib.cos(sub[p]["self"], sub[p]["tgt"]) for p in ps]
        tr_fm = [qlib.cos(np.mean([sub[p]["dons"][m] for m in fm_d], axis=0),
                          sub[p]["tgt"]) for p in ps]
        tr_all = [qlib.cos(np.mean(list(sub[p]["dons"].values()), axis=0),
                           sub[p]["tgt"]) for p in ps]
        # shuffled floor: donor estimate of a random OTHER perturbation
        sh = rng.permutation(len(ps))
        sh = np.where(sh == np.arange(len(ps)), (sh + 1) % len(ps), sh)
        tr_sh = [qlib.cos(np.mean([sub[ps[sh[i]]]["dons"][m] for m in fm_d],
                                  axis=0), sub[ps[i]]["tgt"])
                 for i in range(len(ps))]
        mc, mf, ma, ms = map(lambda v: float(np.mean(v)),
                             (ceil, tr_fm, tr_all, tr_sh))
        out["targets"][t] = {
            "n_perts": len(ps), "ceiling": mc,
            "fm_donors": mf, "fm_pct_ceiling": mf / mc,
            "all_donors": ma, "all_pct_ceiling": ma / mc,
            "shuffled_floor": ms}
        print(f"{t:<7} {mc:8.3f} {mf:10.3f} {mf/mc:6.1%} "
              f"{ma:11.3f} {ma/mc:6.1%} {ms:9.3f}")

    # does adding PCA as a donor help or hurt the FM targets?
    deltas = []
    for t in FM:
        sub = rec[t]; ps = sorted(sub)
        fm_d = [m for m in FM if m != t]
        d = [qlib.cos(np.mean([sub[p]["dons"][m] for m in fm_d + ["pca"]],
                              axis=0), sub[p]["tgt"])
             - qlib.cos(np.mean([sub[p]["dons"][m] for m in fm_d], axis=0),
                        sub[p]["tgt"]) for p in ps]
        deltas.extend(d)
    d = np.array(deltas)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out["pca_as_extra_donor_delta"] = {"delta": float(d.mean()),
                                       "ci": [float(lo), float(hi)]}
    print(f"\nPCA added to FM donor pools: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")

    pct = [out["targets"][t]["fm_pct_ceiling"] for t in FM if t in out["targets"]]
    ppca = out["targets"].get("pca", {}).get("fm_pct_ceiling")
    out["verdict"] = (
        f"PCA target reaches {ppca:.0%} of its ceiling from FM donors "
        f"(FM targets: {min(pct):.0%}-{max(pct):.0%})" if ppca else "pca row missing")
    print(f"VERDICT: {out['verdict']}")
    qlib.save_result(f"e38_pca_member_{qlib.DATASET}", out)
    print("E38_DONE", flush=True)


if __name__ == "__main__":
    main()
