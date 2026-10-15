"""Extract matched scGPT and Geneformer embeddings for the same PBMC cells."""
import sys, pathlib, json, time
import numpy as np, scanpy as sc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import encoders

SP = "/private/tmp/claude-501/-Users-jeffreyxie-Desktop-Silico/0565d63b-a4e0-4a84-9de8-f7151ec02253/scratchpad"
sc.settings.datasetdir = f"{SP}/sc"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

a = sc.datasets.pbmc3k_processed()
raw = a.raw.to_adata()
raw.obs["louvain"] = a.obs["louvain"].values
rng = np.random.default_rng(0)
idx = rng.permutation(raw.n_obs)[:N]
sub = raw[idx].copy()
X = np.asarray(sub.X.todense() if hasattr(sub.X, "todense") else sub.X)
symbols = list(sub.var_names)
labels = sub.obs["louvain"].astype(str).values
print(f"cells {X.shape[0]}, genes {X.shape[1]}, types {len(set(labels))}")
print(f"type counts: {dict(zip(*np.unique(labels, return_counts=True)))}\n")

# Geneformer needs Ensembl IDs; map from symbols using its own name->id dict
import pickle
n2i = pickle.load(open(f"{SP}/models/geneformer/gene_name_id_dict_gc30M.pkl","rb"))
ens = [n2i.get(s, "NA") for s in symbols]
print(f"symbol->ensembl mapped: {sum(e!='NA' for e in ens)}/{len(ens)}\n")

t=time.time(); zg = encoders.embed_geneformer(X, ens, f"{SP}/models/geneformer")
print(f"Geneformer {zg.shape} in {time.time()-t:.0f}s\n")
t=time.time(); zs = encoders.embed_scgpt(X, symbols, f"{SP}/models/scgpt")
print(f"scGPT {zs.shape} in {time.time()-t:.0f}s")

np.savez(f"{SP}/pbmc_embeddings.npz", zs=zs, zg=zg, labels=labels)
print(f"\nsaved -> {SP}/pbmc_embeddings.npz")
print(f"scGPT     norm {np.linalg.norm(zs,axis=1).mean():.3f}  std {zs.std():.4f}")
print(f"Geneformer norm {np.linalg.norm(zg,axis=1).mean():.3f}  std {zg.std():.4f}")
