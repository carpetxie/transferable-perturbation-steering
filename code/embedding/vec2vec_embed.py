"""Recompute every embedding on Modal in one consistent pipeline.

WHY. Embeddings computed on different machines are not comparable: library
versions, device, dtype, batching, dictionary versions and dataset revisions all
move the numbers. This project was bitten once already, when a re-downloaded
scGPT checkpoint reproduced its own quality signatures (probe 0.933 vs 0.920, PR
7.1 vs 7.0) yet moved scGPT->gf2 transfer from 0.532 to 0.632 across all five
seeds, with no error raised.

That matters because the shared space is fitted JOINTLY across every model's
view of THE SAME cells. Mixing pipelines means GCCA aligns things that are not
views of one object, and the result looks plausible while meaning nothing. So
every model is recomputed here: pinned versions, float32, dataset fetched inside
the container.

STAGES
  ping     smoke test that the file, image and volume mount all work
  prepare  fetch datasets (Zenodo) and weights (HF) into the Volume
  stage    sample cells once on CPU, persist only those rows. The GPU path must
           not load the full 111k x 33.7k matrix and convert CSC->CSR, which
           transiently holds two copies.
  embed    one model over one dataset, GPU, checkpointed per condition
  collect  tar the outputs for download

Output filenames match qlib.load_emb(), so downstream analyses run unchanged.
"""
import os
from pathlib import Path

import modal

VOL = modal.Volume.from_name("vec2vec-cache", create_if_missing=True)
CACHE = "/cache"
ENCODERS_SRC = Path(__file__).resolve().parents[1] / "src" / "encoders.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "h5py==3.12.1",
        "huggingface_hub==0.26.2",
        "requests==2.32.3",
        "scvi-tools==1.3.3",
    )
    .add_local_file(ENCODERS_SRC, remote_path="/root/src/encoders.py")
)

app = modal.App("vec2vec-embed", image=image)

ZENODO = "https://zenodo.org/records/13350497/files/{}.h5ad?download=1"
DATASETS = {
    "norman": "NormanWeissman2019_filtered",
    "adamson": "AdamsonWeissman2016_GSM2406681_10X010",
}
GENEFORMERS = {
    "geneformer": ("Geneformer-V1-10M", "gc30M"),
    "geneformer_v2": ("Geneformer-V2-104M", "gc104M"),
    "geneformer_v3": ("Geneformer-V2-316M", "gc104M"),
    "geneformer_cl": ("Geneformer-V2-104M_CLcancer", "gc104M"),
}
CELL_CAP = 100
CTRL_CAP = 6000
MIN_CELLS = 30


def _fit_control_pca(X, n_comp=128, n_hvg=2000, min_detection=0.05):
    """Fit the queue's PCA baseline using control cells only."""
    import numpy as np

    X = X.toarray().astype(np.float64) if hasattr(X, "toarray") \
        else np.asarray(X, dtype=np.float64)
    lib = X.sum(1)
    Xn = np.log1p(X / np.maximum(lib[:, None], 1) * 1e4)
    detected = (X > 0).mean(0)
    eligible = np.flatnonzero(detected >= min_detection)
    if not len(eligible):
        raise ValueError("no genes pass the control-cell detection floor")
    variance = Xn[:, eligible].var(0)
    hvg = eligible[np.argsort(-variance)[:min(n_hvg, len(eligible))]]
    mu = Xn[:, hvg].mean(0)
    _, _, vt = np.linalg.svd(Xn[:, hvg] - mu, full_matrices=False)
    k = min(n_comp, len(vt))
    return hvg, vt[:k].T, mu


def _control_bounds(meta):
    """Return the staged row interval belonging to the designed control."""
    conds = [str(c) for c in meta["conds"]]
    ctrl = str(meta["ctrl"].item())
    i = conds.index(ctrl)
    bounds = meta["bounds"]
    return int(bounds[i]), int(bounds[i + 1])


def _save_condition_embeddings(outdir, model, conds, bounds, Xall, Zall):
    """Persist one qlib-compatible embedding file per condition."""
    import numpy as np

    for i, cond in enumerate(conds):
        start, stop = int(bounds[i]), int(bounds[i + 1])
        f = f"{outdir}/{cond.replace('/', '-')}__{model}.npz"
        if os.path.exists(f):
            continue
        X = Xall[start:stop]
        lib = np.asarray(X.sum(1)).ravel().astype(np.float32)
        ngene = np.asarray((X > 0).sum(1)).ravel().astype(np.float32)
        np.savez(f, Z=Zall[start:stop].astype(np.float32), lib=lib, ngene=ngene)


def _embed_pca_baseline(outdir, meta, Xall):
    """Control-fitted linear representation, promoted to a full model view."""
    import numpy as np

    conds = [str(c) for c in meta["conds"]]
    bounds = meta["bounds"]
    basis_file = f"{outdir}/control_pca_basis.npz"
    if os.path.exists(basis_file):
        d = np.load(basis_file)
        hvg, W, mu = d["hvg"], d["W"], d["mu"]
    else:
        start, stop = _control_bounds(meta)
        hvg, W, mu = _fit_control_pca(Xall[start:stop])
        np.savez(basis_file, hvg=hvg, W=W, mu=mu)
    X = Xall[:, hvg].toarray().astype(np.float64)
    lib = np.asarray(Xall.sum(1)).ravel()
    Xn = np.log1p(X / np.maximum(lib[:, None], 1) * 1e4)
    Z = (Xn - mu) @ W
    _save_condition_embeddings(outdir, "pca", conds, bounds, Xall, Z)
    VOL.commit()
    return f"pca: {len(conds)} conditions, dim={Z.shape[1]}"


def _embed_scvi_baseline(outdir, meta, Xall):
    """Train scVI on controls only, then encode all staged cells label-free."""
    import anndata as ad
    import numpy as np
    import scvi
    import torch

    conds = [str(c) for c in meta["conds"]]
    bounds = meta["bounds"]
    start, stop = _control_bounds(meta)
    hvg_file = f"{outdir}/scvi_hvg.npz"
    if os.path.exists(hvg_file):
        hvg = np.load(hvg_file)["hvg"]
    else:
        hvg, _, _ = _fit_control_pca(Xall[start:stop], n_comp=1)
        np.savez(hvg_file, hvg=hvg)

    var_names = np.array([f"gene_{i}" for i in hvg], dtype=str)
    ctrl = ad.AnnData(X=Xall[start:stop, hvg].copy())
    ctrl.var_names = var_names
    model_dir = f"{outdir}/scvi_model"
    if os.path.exists(f"{model_dir}/model.pt"):
        model = scvi.model.SCVI.load(model_dir, adata=ctrl)
    else:
        scvi.settings.seed = 0
        scvi.model.SCVI.setup_anndata(ctrl)
        model = scvi.model.SCVI(
            ctrl, n_hidden=256, n_latent=64, n_layers=2,
            gene_likelihood="nb", dispersion="gene")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        model.train(
            max_epochs=200, train_size=0.9, validation_size=0.1,
            early_stopping=True, batch_size=256,
            accelerator=accelerator, devices=1,
            enable_progress_bar=True)
        model.save(model_dir, overwrite=True, save_anndata=False)
        VOL.commit()

    all_cells = ad.AnnData(X=Xall[:, hvg].copy())
    all_cells.var_names = var_names
    Z = model.get_latent_representation(adata=all_cells, batch_size=512)
    if not np.isfinite(Z).all():
        raise ValueError("scVI produced non-finite latent values")
    _save_condition_embeddings(outdir, "scvi", conds, bounds, Xall, Z)
    VOL.commit()
    return f"scvi: {len(conds)} conditions, dim={Z.shape[1]}"


def _read_str_column(grp, key):
    import numpy as np
    o = grp[key]
    if hasattr(o, "keys") and "categories" in o:
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        return np.array([cats[i] if 0 <= i < len(cats) else "NA"
                         for i in o["codes"][:]])
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in o[:]])


def _condition_index(path, seed=0):
    """Same sampling logic as the local qlib, so cells match by construction."""
    import collections
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        pert = _read_str_column(f["obs"], "perturbation")
        vi = f["var"].attrs.get("_index", "_index")
        vi = vi.decode() if isinstance(vi, bytes) else str(vi)
        syms = [str(x) for x in _read_str_column(f["var"], vi)]
    cnt = collections.Counter(pert)
    ctrl = "control" if "control" in cnt else max(cnt, key=cnt.get)
    rng = np.random.default_rng(seed)
    idx = {}
    for c in sorted(x for x in cnt if cnt[x] >= MIN_CELLS):
        rows = np.flatnonzero(pert == c)
        cap = CTRL_CAP if c == ctrl else CELL_CAP
        if len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        idx[c] = np.sort(rows)
    return idx, syms, ctrl


def _load_matrix(path):
    """Full count matrix as CSR. Norman is stored CSC, which cannot stream rows."""
    import h5py
    import numpy as np
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        X = f["X"]
        enc = X.attrs.get("encoding-type")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        shape = tuple(int(v) for v in X.attrs["shape"])
        if enc == "csc_matrix":
            return sp.csc_matrix((X["data"][:], X["indices"][:], X["indptr"][:]),
                                 shape=shape).tocsr()
        if enc == "csr_matrix":
            return sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]),
                                 shape=shape)
        return np.asarray(X[:], dtype=np.float32)


@app.function(volumes={CACHE: VOL}, timeout=600)
def ping():
    """Smoke test: same file, image and volume mount as the real functions."""
    import os
    return {"models": sorted(os.listdir(f"{CACHE}/models")),
            "data": sorted(os.listdir(f"{CACHE}/data"))}


@app.function(volumes={CACHE: VOL}, timeout=3600)
def prepare():
    """Fetch datasets and weights into the Volume. Idempotent."""
    import os
    import shutil
    import requests
    from huggingface_hub import hf_hub_download
    os.makedirs(f"{CACHE}/data", exist_ok=True)
    os.makedirs(f"{CACHE}/models", exist_ok=True)

    for key, fn in DATASETS.items():
        dst = f"{CACHE}/data/{fn}.h5ad"
        if os.path.exists(dst):
            print(f"[data] {key} present", flush=True)
            continue
        print(f"[data] fetching {key}", flush=True)
        with requests.get(ZENODO.format(fn), stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)

    d = f"{CACHE}/models/scgpt"
    os.makedirs(d, exist_ok=True)
    for f in ["best_model.pt", "vocab.json", "args.json"]:
        if not os.path.exists(f"{d}/{f}"):
            shutil.copy(hf_hub_download("wanglab/scGPT-human", f), f"{d}/{f}")
    print("[model] scgpt ready", flush=True)

    R = "ctheodoris/Geneformer"
    for name, (sub, gc) in GENEFORMERS.items():
        d = f"{CACHE}/models/{name}"
        os.makedirs(d, exist_ok=True)
        for f in ["config.json", "model.safetensors"]:
            if not os.path.exists(f"{d}/{f}"):
                shutil.copy(hf_hub_download(R, f"{sub}/{f}"), f"{d}/{f}")
        pre = "geneformer/gene_dictionaries_30m/" if gc == "gc30M" else "geneformer/"
        for stem in ["gene_name_id_dict", "gene_median_dictionary", "token_dictionary"]:
            tgt = f"{d}/{stem}_{gc}.pkl"
            if not os.path.exists(tgt):
                shutil.copy(hf_hub_download(R, f"{pre}{stem}_{gc}.pkl"), tgt)
        print(f"[model] {name} ready", flush=True)
    VOL.commit()
    return sorted(os.listdir(f"{CACHE}/models"))


@app.function(volumes={CACHE: VOL}, timeout=3600, memory=16384)
def stage(dataset: str):
    """Sample cells once on CPU; persist only those rows."""
    import os
    import numpy as np
    import scipy.sparse as sp
    os.makedirs(f"{CACHE}/staged", exist_ok=True)
    meta_f = f"{CACHE}/staged/{dataset}.npz"
    if os.path.exists(meta_f):
        print(f"[stage] {dataset} present", flush=True)
        return meta_f
    path = f"{CACHE}/data/{DATASETS[dataset]}.h5ad"
    idx, syms, ctrl = _condition_index(path)
    print(f"[stage] {dataset}: {len(idx)} conditions, control={ctrl}", flush=True)
    M = _load_matrix(path)
    print(f"[stage] matrix {M.shape} loaded", flush=True)
    order = sorted(idx)
    rows = np.concatenate([idx[c] for c in order])
    bounds = np.cumsum([0] + [len(idx[c]) for c in order])
    sub = M[rows]
    sub = sub.tocsr() if sp.issparse(sub) else sp.csr_matrix(sub)
    sp.save_npz(f"{CACHE}/staged/{dataset}_X.npz", sub)
    np.savez(meta_f, conds=np.array(order, dtype=object), bounds=bounds,
             syms=np.array(syms, dtype=object), ctrl=ctrl)
    VOL.commit()
    print(f"[stage] {dataset}: {sub.shape} nnz={sub.nnz}", flush=True)
    return meta_f


@app.function(gpu="H100", volumes={CACHE: VOL}, timeout=7200)
def embed_model(model: str, dataset: str):
    """Embed every condition of one dataset with one model. Checkpointed."""
    import os
    import pickle
    import sys
    import numpy as np
    import scipy.sparse as sp
    import torch
    sys.path.insert(0, "/root/src")
    import encoders

    # encoders.device() picks MPS locally. Force CUDA, keep float32: a
    # half-precision path would move the numbers materially, which is exactly
    # the failure mode this file exists to prevent.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders.device = lambda: dev
    torch.set_grad_enabled(False)
    print(f"start {dataset}/{model} on {dev}", flush=True)

    outdir = f"{CACHE}/emb/{dataset}"
    os.makedirs(outdir, exist_ok=True)
    meta = np.load(f"{CACHE}/staged/{dataset}.npz", allow_pickle=True)
    conds = [str(c) for c in meta["conds"]]
    bounds = meta["bounds"]
    syms = [str(x) for x in meta["syms"]]
    Xall = sp.load_npz(f"{CACHE}/staged/{dataset}_X.npz")
    print(f"  staged {Xall.shape} over {len(conds)} conditions", flush=True)

    if model == "pca":
        return _embed_pca_baseline(outdir, meta, Xall)
    if model == "scvi":
        return _embed_scvi_baseline(outdir, meta, Xall)

    mdir = f"{CACHE}/models/{model}"
    if model != "scgpt":
        gc = GENEFORMERS[model][1]
        n2i = pickle.load(open(f"{mdir}/gene_name_id_dict_{gc}.pkl", "rb"))
        ens = [n2i.get(s, "NA") for s in syms]

    done = 0
    for i, cond in enumerate(conds):
        f = f"{outdir}/{cond.replace('/', '-')}__{model}.npz"
        if os.path.exists(f):
            done += 1
            continue
        X = Xall[bounds[i]:bounds[i + 1]].toarray().astype(np.float32)
        lib = X.sum(1)
        ngene = (X > 0).sum(1).astype(np.float32)
        if model == "scgpt":
            Z = encoders.embed_scgpt(X, syms, mdir, verbose=False)
        else:
            gc = GENEFORMERS[model][1]
            Z = encoders.embed_geneformer(
                X, ens, mdir, max_len=2048, batch=16, verbose=False,
                median_file=f"gene_median_dictionary_{gc}.pkl",
                token_file=f"token_dictionary_{gc}.pkl")
        np.savez(f, Z=Z.astype(np.float32), lib=lib, ngene=ngene)
        done += 1
        if done % 25 == 0:
            VOL.commit()
            print(f"  {done}/{len(conds)}", flush=True)
    VOL.commit()
    return f"{dataset}/{model}: {done} conditions"


@app.function(timeout=900)
def smoke_baselines():
    """Exercise PCA and scVI APIs on synthetic counts before a full run."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import scvi

    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.5, size=(96, 48)).astype(np.float32))
    hvg, W, mu = _fit_control_pca(X[:64], n_comp=8, n_hvg=32,
                                  min_detection=0.0)
    Xn = np.log1p(X[:, hvg].toarray()
                  / np.maximum(np.asarray(X.sum(1)), 1) * 1e4)
    Zp = (Xn - mu) @ W

    ctrl = ad.AnnData(X=X[:64, hvg].copy())
    ctrl.var_names = [f"gene_{i}" for i in hvg]
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(ctrl)
    model = scvi.model.SCVI(ctrl, n_hidden=16, n_latent=4, n_layers=1,
                            gene_likelihood="nb", dispersion="gene")
    model.train(max_epochs=1, train_size=0.9, validation_size=0.1,
                batch_size=32, accelerator="cpu", devices=1,
                enable_progress_bar=False)
    all_cells = ad.AnnData(X=X[:, hvg].copy())
    all_cells.var_names = ctrl.var_names.copy()
    Zs = model.get_latent_representation(adata=all_cells, batch_size=32)
    assert Zp.shape == (96, 8) and np.isfinite(Zp).all()
    assert Zs.shape == (96, 4) and np.isfinite(Zs).all()
    print({"pca": Zp.shape, "scvi": Zs.shape,
           "scvi_tools": scvi.__version__}, flush=True)


@app.function(volumes={CACHE: VOL}, timeout=3600)
def collect(dataset: str):
    """Tar the embeddings for a single download."""
    import os
    import tarfile
    src = f"{CACHE}/emb/{dataset}"
    dst = f"{CACHE}/emb_{dataset}.tar"
    with tarfile.open(dst, "w") as t:
        t.add(src, arcname=dataset)
    VOL.commit()
    return f"{dst} ({os.path.getsize(dst) / 1e6:.0f} MB)"


@app.local_entrypoint()
def main(dataset: str = "norman",
         models: str = "scgpt,geneformer,geneformer_v2,geneformer_v3,geneformer_cl,scvi,pca"):
    print(prepare.remote())
    print(stage.remote(dataset))
    for r in embed_model.starmap([(m, dataset) for m in models.split(",")]):
        print(r)
    print(collect.remote(dataset))
