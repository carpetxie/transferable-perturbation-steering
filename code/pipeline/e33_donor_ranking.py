"""E33: rank donors by their OWN reliability, in a fixed shared space.

THE PROBLEM, stated cleanly.
  given   donor pool D, each WITH labelled perturbation data
          target M, WITHOUT labelled perturbation data
          unlabelled control cells embeddable by every model including M
  find    the subset S of D maximising transfer of perturbation effects into M
  subject to using only control cells and the DONORS' own labels, never M's

TWO FIXES OVER E32.

1. FIXED SPACE. E32 refit GCCA per subset, so every row had a different
   subspace and a different noise ceiling (0.700-0.724 for one target). That
   entangles "which donors contribute deltas" with "which models define the
   space". Here the space is fitted ONCE on all models plus the target's
   control cells, then held fixed while only the donor set varies. Transfer
   numbers become directly comparable, and it matches the operational setting:
   you onboard a target into an existing space, you do not rebuild the space
   per query.

2. A CRITERION THAT IS NOT CONFOUNDED. E32's gcca_corr appeared to beat the
   trivial baseline (mean rho +0.491 vs +0.205) but inspection showed why it
   fails: it measures SIMILARITY to the target, while transfer is governed by
   donor QUALITY. For a Geneformer target the most similar donors are other
   Geneformers, but the best donor is scGPT, the most architecturally distant,
   which had the LOWEST gcca of all seven subsets while giving the HIGHEST
   transfer. gcca_corr only worked for the one target where similarity and
   quality happened to coincide.

   The replacement uses the asymmetry the problem already grants: donors have
   labels. So each donor's own split-half reliability of its effect estimates
   is computable without ever touching the target's labels.

CRITERIA COMPARED (all label-free with respect to the TARGET):
  self_rel     mean split-half reliability of the donor's own effect estimates,
               averaged over perturbations. Pure donor quality.
  gcca_corr    E32's criterion, carried over so the confound is visible.
  self_x_gcca  product of the two: quality weighted by overlap with the target.
  n_donors     trivial baseline.

VALIDATION. Spearman between criterion and achieved transfer, computed WITHIN
each target over its 7 subsets, then averaged. Also reported: whether the
criterion's top-ranked subset is the actually-best subset (top-1 hit rate),
which is what a user of the method would care about.
"""
import itertools
import numpy as np
import qlib

ENC = ["scgpt", "gf1", "gf2", "gf3"]
K = 24
MIN_CELLS = 60


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / d) if d > 0 else 0.0


def main():
    idx, syms, _ = qlib.condition_index()
    C = qlib.CONTROL

    betas, mus, ctrl = {}, {}, {}
    for m in ENC:
        Z, lib, ng = qlib.load_emb(C, m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); ctrl[m] = Zd - mus[m]

    # ---- ONE shared space, all models, control cells only -----------------
    g = qlib.fit_gcca(ctrl, k=K)
    perts = [c for c in sorted(idx.keys() - {C}) if len(idx[c]) >= MIN_CELLS]
    print(f"fixed space: all {len(ENC)} models, K={K}, {len(perts)} perturbations",
          flush=True)

    def S(cond, m):
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ g["P"][m]) @ g["W"][m] @ g["C"][m]

    S0 = {m: S(C, m) for m in ENC}
    Sp = {p: {m: S(p, m) for m in ENC} for p in perts}

    # ---- donor self-reliability: split-half of its OWN effect estimates ----
    # Uses only the donor's labels. The target is never touched.
    self_rel = {}
    for m in ENC:
        v = []
        for p in perts:
            Z = Sp[p][m]
            h = len(Z) // 2
            if h < 10:
                continue
            v.append(qlib.cos(Z[:h].mean(0) - S0[m].mean(0),
                              Z[h:].mean(0) - S0[m].mean(0)))
        self_rel[m] = float(np.mean(v))
    print("\ndonor self-reliability (computable without any target labels):")
    for m, v in sorted(self_rel.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<7} {v:.3f}")

    # ---- transfer for every (target, subset), FIXED space ------------------
    rows = []
    for tgt in ENC:
        pool = [m for m in ENC if m != tgt]
        # target-side overlap with each donor, control cells only
        Tc = S(C, tgt)
        for r_ in range(1, len(pool) + 1):
            for subset in itertools.combinations(pool, r_):
                cons = np.mean([S(C, m) for m in subset], axis=0)
                n = min(len(Tc), len(cons))
                cc = np.nan_to_num(np.array(
                    [abs(np.corrcoef(Tc[:n, j], cons[:n, j])[0, 1])
                     for j in range(K)]))
                tr, ce = [], []
                for p in perts:
                    Zt = Sp[p][tgt]
                    h = len(Zt) // 2
                    if h < 10:
                        continue
                    gold = Zt[h:].mean(0) - S0[tgt].mean(0)
                    ce.append(qlib.cos(Zt[:h].mean(0) - S0[tgt].mean(0), gold))
                    d = np.mean([Sp[p][m][:len(Sp[p][m]) // 2].mean(0)
                                 - S0[m].mean(0) for m in subset], axis=0)
                    tr.append(qlib.cos(d, gold))
                sr = float(np.mean([self_rel[m] for m in subset]))
                rows.append({
                    "target": tgt, "subset": list(subset), "n_donors": len(subset),
                    "transfer": float(np.mean(tr)), "ceiling": float(np.mean(ce)),
                    "self_rel": sr, "gcca_corr": float(cc.mean()),
                    "self_x_gcca": sr * float(cc.mean())})

    print(f"\n{'target':<7} {'donors':<20} {'transfer':>9} {'self_rel':>9} {'gcca':>7}")
    for tgt in ENC:
        rs = sorted([r for r in rows if r["target"] == tgt],
                    key=lambda r: -r["transfer"])
        for r in rs:
            print(f"{tgt:<7} {'+'.join(r['subset']):<20} {r['transfer']:9.3f} "
                  f"{r['self_rel']:9.3f} {r['gcca_corr']:7.3f}")
        print(f"{'':<7} {'(ceiling)':<20} {rs[0]['ceiling']:9.3f}")

    # ---- does each criterion rank subsets correctly? -----------------------
    print("\n" + "=" * 72)
    print("criterion validation (Spearman within target, then averaged)")
    print(f"{'criterion':<14} {'mean rho':>9}  {'per-target':<28} {'top-1 hit':>10}")
    out = {"self_rel_per_model": self_rel, "rows": rows, "criteria": {}}
    for crit in ["self_rel", "gcca_corr", "self_x_gcca", "n_donors"]:
        rhos, hits = [], 0
        for tgt in ENC:
            rs = [r for r in rows if r["target"] == tgt]
            rhos.append(spearman([r[crit] for r in rs], [r["transfer"] for r in rs]))
            pick = max(rs, key=lambda r: r[crit])
            best = max(rs, key=lambda r: r["transfer"])
            hits += int(pick["subset"] == best["subset"])
        out["criteria"][crit] = {"mean_rho": float(np.mean(rhos)),
                                 "per_target": rhos, "top1_hits": hits}
        tag = "  <- trivial" if crit == "n_donors" else ""
        print(f"{crit:<14} {np.mean(rhos):+9.3f}  "
              f"{str([round(x,2) for x in rhos]):<28} {hits}/{len(ENC)}{tag}")

    # ---- the practical rule: pick the single best-self_rel donor -----------
    print("\npractical rule: 'use the single donor with highest self-reliability'")
    gains = []
    for tgt in ENC:
        rs = [r for r in rows if r["target"] == tgt]
        singles = [r for r in rs if r["n_donors"] == 1]
        pick = max(singles, key=lambda r: r["self_rel"])
        best = max(rs, key=lambda r: r["transfer"])
        allr = [r for r in rs if r["n_donors"] == len(ENC) - 1][0]
        gains.append(pick["transfer"] - allr["transfer"])
        print(f"  {tgt:<7} rule picks {'+'.join(pick['subset']):<8} "
              f"{pick['transfer']:.3f} | oracle best {'+'.join(best['subset']):<12} "
              f"{best['transfer']:.3f} | use-all {allr['transfer']:.3f} "
              f"| rule-vs-all {pick['transfer']-allr['transfer']:+.3f}")
    out["rule_vs_useall_mean"] = float(np.mean(gains))
    print(f"  mean gain of rule over using every donor: {np.mean(gains):+.3f}")

    best_c = max(["self_rel", "gcca_corr", "self_x_gcca"],
                 key=lambda c: out["criteria"][c]["mean_rho"])
    v = (f"best criterion '{best_c}' rho={out['criteria'][best_c]['mean_rho']:+.3f} "
         f"vs trivial {out['criteria']['n_donors']['mean_rho']:+.3f}; "
         f"rule beats use-all by {np.mean(gains):+.3f}")
    out["verdict"] = v
    print(f"\nVERDICT: {v}", flush=True)
    qlib.save_result("e33_donor_ranking", out)
    print("E33_DONE", flush=True)


if __name__ == "__main__":
    main()
