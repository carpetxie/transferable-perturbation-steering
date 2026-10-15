"""Do the shared directions carry NAMED biological programs, or do I just see faces in clouds?

interpret.py showed that after dimension 1 (which is technical -- |r| 0.91-0.96
with genes-detected), dimensions 2-5 have gene loadings that LOOK like cell
cycle, unfolded protein response, and interferon. Reading biology off the top-8
gene names is how people fool themselves: any 8 genes suggest a story, and the
highly expressed genes that dominate loadings are exactly the ones that appear
in every curated set.

So each direction is scored against curated sets with a null that controls for
the two things that manufacture false hits:

    size          the null set has the same number of genes
    expression    the null genes are drawn from the same mean-expression deciles,
                  because highly expressed genes have higher correlations with
                  almost anything

Score = mean |correlation with the canonical variate| over the set's genes.
Reported as a z-score against 300 matched random sets. A program counts as
present only if the real set beats every null draw (empirical p < 1/300).
"""
import sys, json
import numpy as np
sys.path.insert(0,"src")
from sweep import embed_all, load_tissue
from interpret import variates, corr

SETS = {
 "cell_cycle_G2M": """BIRC5 BUB1 CCNB2 CDC20 CDCA3 CDCA8 CDK1 CENPA CENPE CENPF
   CKS1B CKS2 KIF11 KIF20B KIF23 MKI67 NUSAP1 PLK1 TOP2A TPX2 TUBB4B UBE2C
   ANLN AURKA AURKB CDCA2 GTSE1 HMGB2 NDC80 SMC4 TACC3 CDKN3 CCNA2""".split(),
 "cell_cycle_S": """ATAD2 BLM BRIP1 CASP8AP2 CCNE2 CDC45 CDC6 CDCA7 CHAF1B CLSPN
   DSCC1 DTL E2F8 EXO1 FEN1 GINS2 GMNN HELLS MCM2 MCM4 MCM5 MCM6 MSH2 NASP
   PCNA POLA1 PRIM1 RFC2 RRM1 RRM2 SLBP TIPIN TYMS UBR7 UHRF1 USP1 WDR76""".split(),
 "UPR_ER_stress": """HSPA5 DDIT3 XBP1 PDIA4 PDIA6 HYOU1 DNAJB9 MANF SDF2L1 TRIB3
   ATF4 HERPUD1 EDEM1 SEC61A1 CALR CANX P4HB PDIA3 DNAJC3 SEL1L SRPRB CRELD2
   HSP90B1 ERO1A ATF3 NUPR1""".split(),
 "interferon": """ISG15 IFI6 IFIT1 IFIT2 IFIT3 MX1 MX2 OAS1 OAS2 OAS3 STAT1 STAT2
   IRF7 IRF9 TAP1 ISG20 WARS1 WARS BST2 IFI44L IFI44 IFITM1 IFITM2 IFITM3 B2M
   HLA-A HLA-B HLA-C PSMB8 PSMB9 XAF1 SAMD9L""".split(),
 "ribosome": None,      # filled per tissue from gene names
}

def matched_null(idx, mean_expr, rng, n_null=300):
    """Random sets matched on size and mean-expression decile."""
    dec = np.digitize(mean_expr, np.quantile(mean_expr, np.linspace(0,1,11)[1:-1]))
    pools = {d: np.flatnonzero(dec==d) for d in np.unique(dec)}
    want = [int(dec[i]) for i in idx]
    out=[]
    for _ in range(n_null):
        out.append(np.array([rng.choice(pools[d]) for d in want]))
    return out

res={}
for t in ["thp1","neuron","pbmc"]:
    X,syms,lab = load_tissue(t,1000); X=np.asarray(X,dtype=np.float64)
    lib=X.sum(1); Xn=np.log1p(X/np.maximum(lib[:,None],1)*1e4)
    mean_expr=Xn.mean(0)
    up=[str(s).upper() for s in syms]; pos={s:i for i,s in enumerate(up)}
    sets=dict(SETS); sets["ribosome"]=[s for s in up if s.startswith(("RPS","RPL"))][:40]
    E=embed_all(t,1000)
    print(f"\n=== {t} ===")
    for a,b in [("scgpt","gf2"),("gf1","gf2")]:
        V=variates(E[a].astype(np.float64),E[b].astype(np.float64),k=5)
        key=f"{t}:{a}->{b}"; res[key]={}
        for d in range(V.shape[1]):
            g=np.abs(np.array([corr(V[:,d],Xn[:,j]) for j in range(Xn.shape[1])]))
            line=[]
            for nm,genes in sets.items():
                idx=[pos[s] for s in genes if s in pos]
                if len(idx)<8: continue
                rng=np.random.default_rng(7)
                real=g[idx].mean()
                nl=np.array([g[i].mean() for i in matched_null(idx,mean_expr,rng)])
                z=(real-nl.mean())/max(nl.std(),1e-9)
                beats=bool(real>nl.max())
                res[key].setdefault(f"dim{d+1}",{})[nm]={
                    "z":float(z),"real":float(real),"null_mean":float(nl.mean()),
                    "beats_all_nulls":beats,"n_genes":len(idx)}
                if beats and z>3: line.append(f"{nm}(z={z:.1f})")
            print(f"  {a}->{b} dim{d+1}: {', '.join(line) if line else '-'}")
json.dump(res,open("artifacts/genesets.json","w"),indent=2,default=float)
hits=[(k,d,n,v["z"]) for k,dd in res.items() for d,ss in dd.items()
      for n,v in ss.items() if v["beats_all_nulls"] and v["z"]>3]
print(f"\n{len(hits)} program hits beating all 300 matched nulls at z>3")
