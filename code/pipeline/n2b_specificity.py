"""Is the cross-model transfer PERTURBATION-SPECIFIC, or just generic treatment?

n2 shows a transferred effect matches the held-out model's measured effect at
cos 0.531 (matched budget), against 0.240 for shuffled labels. That 0.240 floor
is the Systema critique made concrete: perturbations share a generic
treated-versus-control direction, so a high cosine can be mostly generic.

The decisive test is retrieval, not cosine. Transfer perturbation p's effect
into held-out model m, then rank ALL perturbations by how well the transferred
vector matches each one's MEASURED effect in m. If the correct perturbation
ranks first, the transferred signal carries perturbation identity, not just
"something was perturbed".

Chance is 1/201. A generic-only signal would rank near chance while still
scoring a high raw cosine.

CONTROL: the same retrieval after subtracting the mean effect across all
perturbations from every vector, which removes the generic axis entirely. If
retrieval survives that, the specificity is not carried by the generic
direction at all.
"""
import numpy as np, qlib
import n2_equivariance as n2mod

MODELS = ["scgpt", "gf1", "gf2"]
K = 24

def main():
    idx, syms, triples = qlib.condition_index()
    betas, mus, views = {}, {}, {}
    for m in MODELS:
        Z, lib, ng = qlib.load_emb("control", m)
        b, _ = qlib.fit_depth_projector(Z, lib, ng)
        Zd = qlib.apply_depth_projector(Z, lib, ng, b)
        betas[m] = b; mus[m] = Zd.mean(0); views[m] = Zd - mus[m]
    g = qlib.fit_gcca(views, k=K)
    def S_of(cond, m):
        Z, lib, ng = qlib.load_emb(cond, m)
        Zd = qlib.apply_depth_projector(Z, lib, ng, betas[m]) - mus[m]
        return (Zd @ g["P"][m]) @ g["W"][m] @ g["C"][m]

    perts = [c for c in sorted(idx.keys() - {"control"}) if len(idx[c]) >= 60]
    S0 = {m: S_of("control", m) for m in MODELS}
    D = {m: {} for m in MODELS}
    for p in perts:
        for m in MODELS:
            D[m][p] = S_of(p, m).mean(0) - S0[m].mean(0)

    res = {}
    for centered in (False, True):
        ranks_all = []
        for m_out in MODELS:
            m_in = [m for m in MODELS if m != m_out]
            meas = np.stack([D[m_out][p] for p in perts])
            pred = np.stack([np.mean([D[m][p] for m in m_in], axis=0) for p in perts])
            if centered:
                meas = meas - meas.mean(0)
                pred = pred - pred.mean(0)
            mn = meas / (np.linalg.norm(meas, axis=1, keepdims=True) + 1e-12)
            pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
            Sm = pn @ mn.T
            order = np.argsort(-Sm, axis=1)
            r = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(len(perts))])
            ranks_all.append(r)
        r = np.concatenate(ranks_all)
        res["centered" if centered else "raw"] = {
            "top1": float((r == 1).mean()), "top5": float((r <= 5).mean()),
            "mean_rank": float(r.mean()), "n": int(len(r)),
            "chance_top1": 1.0 / len(perts)}

    print(f"{len(perts)} perturbations, chance top-1 = {1/len(perts):.4f}\n")
    print(f"{'variant':<10} {'top-1':>8} {'top-5':>8} {'mean rank':>10}")
    for k, v in res.items():
        print(f"{k:<10} {v['top1']:8.3f} {v['top5']:8.3f} {v['mean_rank']:10.1f}")
    print(f"\nfold-enrichment over chance: raw {res['raw']['top1']*len(perts):.0f}x, "
          f"generic-removed {res['centered']['top1']*len(perts):.0f}x")
    qlib.save_result("n2b_specificity", res)
    print("N2B_DONE", flush=True)

if __name__ == "__main__":
    main()
