"""Experiment 4: where does linearity break, off the control manifold?

THE QUESTION. Every steering/transport idea in this queue assumes the linear map
holds where you are steering TO. But the maps were fitted on cells that exist.
This measures transport fidelity as a function of distance from the training
distribution, producing a trust radius.

DESIGN.
  - Fit T: scgpt -> gf2 on CONTROL cells only (ridge, CV-tuned alpha).
  - Apply to held-out PERTURBED cells, which the fit has never seen and which
    are displaced from the control manifold by construction.
  - Bin perturbed cells by displacement (Mahalanobis-like distance from the
    control centroid in the control PCA metric) and report per-bin fidelity:
    cosine between T(z_scgpt) and the true z_gf2, plus top-1 retrieval within
    the bin's own gallery.
  - Inside each bin, run the MLP with the same protocol. Finding 1's 57/57 is a
    global result; the steering question is whether linear stays ahead FAR from
    the data or only wins near it where density regularises the fit.
  - Training-size sweep: n_train = 500, 1500, 5000 controls. If the MLP crosses
    over at large n, the linearity headline is a statement about estimation
    under sample constraints, not about the map itself.

DEPTH. Displacement is measured AFTER depth projection, and the depth projector
is fitted on controls only. Without this the experiment reduces to measuring
depth extrapolation: doubles systematically differ in library size, so raw
distance from the control centroid is substantially a depth axis.

CONTROLS.
  - Fidelity on held-out CONTROL cells is the ceiling (bin 0): transport error
    with zero extrapolation.
  - A shuffled-pair floor: cosine between T(z) of one cell and the true z_gf2
    of a DIFFERENT cell in the same bin, which is what fidelity looks like when
    transport carries no per-cell information.
"""
import numpy as np
import qlib


def mlp_map(A_tr, B_tr, A_te, seed=0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(A_tr)
    m = MLPRegressor(hidden_layer_sizes=(512, 512), max_iter=600,
                     early_stopping=True, random_state=seed)
    m.fit(sc.transform(A_tr), B_tr)
    return m.predict(sc.transform(A_te))


def main():
    idx, syms, triples = qlib.condition_index()
    pert_conds = sorted(c for c in idx if c != "control")

    # ---- control embeddings + depth projectors (fitted on train controls only,
    # per split, so no held-out information enters any fit)
    E1, E2 = qlib.PAIR_E1, qlib.PAIR_E2
    Za_c, lib_c, ng_c = qlib.load_emb("control", E1)
    Zb_c, _, _ = qlib.load_emb("control", E2)

    # ---- perturbed pool (subsample for tractability; seeded)
    rng = np.random.default_rng(0)
    per_cond = 40
    Pa, Pb, Plib, Png, Pcond = [], [], [], [], []
    for c in pert_conds:
        Za, lib, ng = qlib.load_emb(c, E1)
        Zb, _, _ = qlib.load_emb(c, E2)
        take = rng.permutation(len(Za))[:per_cond]
        Pa.append(Za[take]); Pb.append(Zb[take])
        Plib.append(lib[take]); Png.append(ng[take])
        Pcond += [c] * len(take)
    Pa, Pb = np.vstack(Pa), np.vstack(Pb)
    Plib, Png = np.concatenate(Plib), np.concatenate(Png)
    print(f"perturbed pool: {len(Pa)} cells from {len(pert_conds)} conditions",
          flush=True)

    N_BINS = 5
    out = {"bins": N_BINS, "E1": E1, "E2": E2, "sweeps": {}}
    for n_train in (500, 1500, 5000):
        p = np.random.default_rng(1).permutation(len(Za_c))
        tr, ev = p[:n_train], p[n_train:n_train + 800]

        # depth projector on TRAIN controls only
        beta, _ = qlib.fit_depth_projector(Za_c[tr], lib_c[tr], ng_c[tr])
        A_tr = qlib.apply_depth_projector(Za_c[tr], lib_c[tr], ng_c[tr], beta)
        A_ev = qlib.apply_depth_projector(Za_c[ev], lib_c[ev], ng_c[ev], beta)
        A_pe = qlib.apply_depth_projector(Pa, Plib, Png, beta)
        # target space depth-projected with its own train-control projector
        betaB, _ = qlib.fit_depth_projector(Zb_c[tr], lib_c[tr], ng_c[tr])
        B_tr = qlib.apply_depth_projector(Zb_c[tr], lib_c[tr], ng_c[tr], betaB)
        B_ev = qlib.apply_depth_projector(Zb_c[ev], lib_c[ev], ng_c[ev], betaB)
        B_pe = qlib.apply_depth_projector(Pb, Plib, Png, betaB)

        # standardise by TRAIN stats
        muA, sdA = A_tr.mean(0), A_tr.std(0) + 1e-6
        muB, sdB = B_tr.mean(0), B_tr.std(0) + 1e-6
        sa = lambda Z: (Z - muA) / sdA
        sb = lambda Z: (Z - muB) / sdB
        A_tr, A_ev, A_pe = sa(A_tr), sa(A_ev), sa(A_pe)
        B_tr, B_ev, B_pe = sb(B_tr), sb(B_ev), sb(B_pe)

        # displacement metric: whitened distance from control centroid in the
        # train-control PCA metric (top 50 PCs, variance-normalised)
        U, S, Vt = np.linalg.svd(A_tr - A_tr.mean(0), full_matrices=False)
        k = min(50, len(S))
        proj = Vt[:k].T / (S[:k] / np.sqrt(len(A_tr)))
        dist_pe = np.linalg.norm((A_pe - A_tr.mean(0)) @ proj, axis=1)
        dist_ev = np.linalg.norm((A_ev - A_tr.mean(0)) @ proj, axis=1)

        # maps fitted on train controls
        W, alpha = qlib.ridge_map(A_tr, B_tr, seed=0)
        lin_pe, lin_ev = A_pe @ W, A_ev @ W
        mlp_pe = mlp_map(A_tr, B_tr, A_pe)
        mlp_ev = mlp_map(A_tr, B_tr, A_ev)

        # bin edges from perturbed-cell distance quantiles
        edges = np.quantile(dist_pe, np.linspace(0, 1, N_BINS + 1))
        rows = []

        def fid(P, T):
            return float(np.mean([qlib.cos(P[i], T[i]) for i in range(len(P))]))

        def shuffled_fid(P, T, rng_):
            j = rng_.permutation(len(T))
            return float(np.mean([qlib.cos(P[i], T[j[i]]) for i in range(len(P))]))

        # bin 0: held-out controls (zero extrapolation ceiling)
        r0 = {"bin": 0, "label": "held-out controls",
              "median_dist": float(np.median(dist_ev)), "n": len(A_ev),
              "linear_cos": fid(lin_ev, B_ev), "mlp_cos": fid(mlp_ev, B_ev),
              "floor_cos": shuffled_fid(lin_ev, B_ev, np.random.default_rng(2))}
        rows.append(r0)
        for b in range(N_BINS):
            m = (dist_pe >= edges[b]) & (dist_pe <= edges[b + 1])
            if m.sum() < 50:
                continue
            rng_ = np.random.default_rng(3 + b)
            rows.append({
                "bin": b + 1, "label": f"perturbed q{b+1}",
                "median_dist": float(np.median(dist_pe[m])), "n": int(m.sum()),
                "linear_cos": fid(lin_pe[m], B_pe[m]),
                "mlp_cos": fid(mlp_pe[m], B_pe[m]),
                "floor_cos": shuffled_fid(lin_pe[m], B_pe[m], rng_)})
        out["sweeps"][str(n_train)] = {"alpha": alpha, "rows": rows}
        print(f"\nn_train={n_train} (ridge alpha {alpha:.1f})", flush=True)
        print(f"{'bin':<20} {'med dist':>9} {'n':>6} {'linear':>8} {'mlp':>8} {'floor':>8}")
        for r in rows:
            print(f"{r['label']:<20} {r['median_dist']:9.2f} {r['n']:6d} "
                  f"{r['linear_cos']:8.3f} {r['mlp_cos']:8.3f} {r['floor_cos']:8.3f}",
                  flush=True)

    qlib.save_result("q4_trust_radius", out)
    print("Q4_DONE", flush=True)


if __name__ == "__main__":
    main()
