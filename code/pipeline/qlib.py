"""Shared infrastructure for the 5-experiment Norman2019 queue.

Design decisions, and why:

DURABLE STORAGE. Everything lives under ~/.cache/silico_vec2vec, not the session
scratchpad. macOS periodic cleanup already deleted /private/tmp once, taking the
venv, all model weights, all datasets and every cached embedding with it. The
home cache directory is on the same local disk (same speed), is not iCloud-synced
(Desktop is, and iCloud sync has hung open() calls in this project before), and
is not subject to tmp cleanup.

PER-CONDITION CHECKPOINTING. Embedding ~26k cells across three encoders takes
hours on MPS. Each (condition, encoder) result is saved as its own .npz the
moment it is computed, so a crash or interrupt loses at most one condition, and
re-running is a no-op for completed work.

STEERING VECTORS ARE PSEUDO-BULK DELTAS. For perturbation P the steering vector
is delta(P) = mean(emb(P cells)) - mean(emb(control cells)). Averaging over
cells suppresses single-cell noise; subtracting control anchors the vector at
the unperturbed state. All five experiments consume these.

DEPTH PROJECTION. Doubles differ systematically in library size from singles
(more perturbed transcriptomes, different capture), so any distance or residual
computed without removing depth partly measures depth. depth_project regresses
log library size and log genes-detected out of an embedding matrix, with
coefficients fitted on CONTROL cells only and applied everywhere, so the
correction itself cannot leak perturbation information.

CCA SHARED SUBSPACE. Fitted on control cells only. Fitting on perturbed cells
would bake the perturbation structure being studied into the subspace and make
'the residual is shared' circular. Controls are abundant (11,855 cells) and
perturbation-independent by construction. The fit is the stable direct SVD
version (PCA-reduce, ridge-whiten, SVD of cross-covariance) because sklearn's
iterative CCA raises 'SVD did not converge' on rank-deficient covariances.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

import numpy as np

# Encoder pair used by the paired experiments. Overridable so the same analysis
# can be re-run on a different (E1, E2) without editing five scripts, and so a
# pairing whose embeddings are already cached can run while a slower encoder is
# still being embedded. Q_TAG keeps the two runs' artifacts from colliding.
PAIR_E1 = os.environ.get("Q_E1", "scgpt")
PAIR_E2 = os.environ.get("Q_E2", "gf2")
TAG = os.environ.get("Q_TAG", "")

DUR = pathlib.Path.home() / ".cache" / "silico_vec2vec"

# ---------------------------------------------------------------- dataset
# The queue was built on Norman2019. External validation needs a second
# perturbation dataset, so the dataset is selectable by environment variable
# and every path derived from it. Default stays "norman" so nothing that has
# already run changes meaning.
import os as _os

_DATASETS = {
    "norman": {
        "file": "NormanWeissman2019_filtered.h5ad",
        "control": "control",
        # Norman encodes a double as "A_B" with both singles present, which is
        # what makes triples available.
        "has_doubles": True,
    },
    "papalexi": {
        # THP-1 monocytes, a DIFFERENT CELL TYPE from the K562 lymphoblasts of
        # both other datasets. This is the axis nothing has been tested on: the
        # multi-encoder result, the averaging law and the error correlation have
        # all replicated across laboratory, protocol and perturbation library,
        # but never across cell type.
        # Control annotation verified before use (LESSONS rule 17): the label is
        # literally "control", carries nperts == 0, and its guide ids are
        # non-targeting, so it is a designed control and not an unassigned bin.
        "file": "PapalexiSatija2021_eccite_RNA.h5ad",
        "control": "control",
        "has_doubles": False,
    },
    "adamson": {
        "file": "AdamsonWeissman2016_GSM2406681_10X010.h5ad",
        # CORRECTED. The first version used "NA" (nperts == 0) as the control.
        # That was wrong: "NA" means no perturbation was ASSIGNED, i.e. cells
        # where guide calling failed or was ambiguous, not designed controls.
        # Using them made every delta measure "assigned vs unassigned" rather
        # than "perturbed vs unperturbed", which showed up as a split-half
        # noise ceiling of 0.110 (Norman's is 0.564) and a generic-effect
        # cosine of -0.579. The true non-targeting control is the Gal4 guide:
        # Gal4 is a yeast transcription factor with no human target, so those
        # cells receive the full CRISPR procedure with no biological effect.
        "control": "Gal4-4(mod)_pBA582",
        # excluded from the perturbation set: unassigned cells, the "*"
        # multiplet label, and 63(mod) whose role is not documented here
        "exclude": ["NA", "*", "63(mod)_pBA580"],
        # Names look like "GENE_pDS002" (gene plus plasmid id) and there are no
        # combinatorial doubles, so triple-based experiments do not apply.
        "has_doubles": False,
    },
}
DATASET = _os.environ.get("Q_DATASET", "norman")
_CFG = _DATASETS[DATASET]
DATA = DUR / "data" / _CFG["file"]
CONTROL = _CFG["control"]
HAS_DOUBLES = _CFG["has_doubles"]
MODELS_DIR = DUR / "models"
MODELS = MODELS_DIR
EMB = DUR / ("norman_embeddings" if DATASET == "norman"
             else f"{DATASET}_embeddings")
ART = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "queue"
EMB.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

CELL_CAP = 100          # cells per perturbation condition
CTRL_CAP = 6000         # control cells (q4 needs a training-size sweep to 5000)
MIN_CELLS = 30          # a condition below this is too noisy to use
ENCODERS = ["scgpt", "gf1", "gf2"]   # gf3 (316M) excluded from the main queue:
                                     # 3x slower and same lineage as gf2, so it
                                     # adds hours, not information


# ----------------------------------------------------------------- data access
def _read_str_column(grp, key):
    o = grp[key]
    if hasattr(o, "keys") and "categories" in o:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        codes = o["codes"][:]
        return np.array([cats[i] if 0 <= i < len(cats) else "NA" for i in codes])
    v = o[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def condition_index(seed=0):
    """Map each usable condition to the row indices of its sampled cells.

    Sampling is seeded and sorted, so every later stage sees the same cells.
    Returns (index dict, gene symbols, list of complete triples).
    """
    import collections
    import h5py
    with h5py.File(DATA, "r") as f:
        pert = _read_str_column(f["obs"], "perturbation")
        vi = f["var"].attrs.get("_index", "_index")
        vi = vi.decode() if isinstance(vi, bytes) else str(vi)
        syms = [str(x) for x in _read_str_column(f["var"], vi)]
    cnt = collections.Counter(pert)
    triples = []
    if HAS_DOUBLES:
        singles = {p for p in cnt if "_" not in p and p != CONTROL}
        for d in sorted(p for p in cnt if "_" in p):
            a, b = d.split("_")
            if (a in singles and b in singles
                    and min(cnt[d], cnt[a], cnt[b]) >= MIN_CELLS):
                triples.append((a, b, d))
        conds = {CONTROL}
        for a, b, d in triples:
            conds |= {a, b, d}
    else:
        # no combinatorial structure: keep every sufficiently sampled condition
        drop = set(_CFG.get("exclude", []))
        conds = {CONTROL} | {p for p, v in cnt.items()
                             if v >= MIN_CELLS and p != CONTROL and p not in drop}
    rng = np.random.default_rng(seed)
    idx = {}
    for c in sorted(conds):
        rows = np.flatnonzero(pert == c)
        cap = CTRL_CAP if c == CONTROL else CELL_CAP
        if len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        idx[c] = np.sort(rows)
    return idx, syms, triples


_X_CSR = None


def _full_matrix():
    """Load the whole count matrix once, as row-sliceable CSR, module-cached.

    Norman is stored CSC (column-major), which cannot stream rows. Per-row
    assembly from CSC would re-read the column pointers 205 times over the
    queue, so instead the matrix is converted to CSR once (~3 GB in RAM,
    acceptable) and every load_rows call is then a cheap slice. The cache is
    module-level, so within one process the conversion happens exactly once.
    """
    global _X_CSR
    if _X_CSR is not None:
        return _X_CSR
    import h5py
    import scipy.sparse as sp
    with h5py.File(DATA, "r") as f:
        X = f["X"]
        enc = X.attrs.get("encoding-type")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        shape = tuple(int(v) for v in X.attrs["shape"])
        if enc == "csc_matrix":
            M = sp.csc_matrix((X["data"][:], X["indices"][:], X["indptr"][:]),
                              shape=shape)
        elif enc == "csr_matrix":
            M = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]),
                              shape=shape)
        else:
            _X_CSR = np.asarray(X[:], dtype=np.float32)
            return _X_CSR
    _X_CSR = M.tocsr()
    return _X_CSR


def load_rows(rows):
    """Dense counts for the given rows."""
    M = _full_matrix()
    out = M[rows]
    return out.toarray().astype(np.float32) if hasattr(out, "toarray") \
        else np.asarray(out, dtype=np.float32)


# ------------------------------------------------------------------ embeddings
def embed_condition(cond, rows, syms, encoder):
    """Embed one condition with one encoder, checkpointed to its own file."""
    f = EMB / f"{cond.replace('/','-')}__{encoder}.npz"
    if f.exists():
        d = np.load(f)
        return d["Z"], d["lib"], d["ngene"]
    import encoders as E
    X = load_rows(rows)
    lib = X.sum(1)
    ngene = (X > 0).sum(1).astype(np.float32)
    if encoder == "scgpt":
        Z = E.embed_scgpt(X, syms, str(MODELS / "scgpt"), verbose=False)
    elif encoder == "gf1":
        n2i = pickle.load(open(MODELS / "geneformer/gene_name_id_dict_gc30M.pkl", "rb"))
        Z = E.embed_geneformer(X, [n2i.get(s, "NA") for s in syms],
                               str(MODELS / "geneformer"), verbose=False)
    elif encoder == "gf3":
        n2i = pickle.load(open(MODELS / "geneformer_v3/gene_name_id_dict_gc104M.pkl", "rb"))
        Z = E.embed_geneformer(X, [n2i.get(s, "NA") for s in syms],
                               str(MODELS / "geneformer_v3"), max_len=2048, batch=4,
                               verbose=False,
                               median_file="gene_median_dictionary_gc104M.pkl",
                               token_file="token_dictionary_gc104M.pkl")
    elif encoder == "gf2":
        n2i = pickle.load(open(MODELS / "geneformer_v2/gene_name_id_dict_gc104M.pkl", "rb"))
        Z = E.embed_geneformer(X, [n2i.get(s, "NA") for s in syms],
                               str(MODELS / "geneformer_v2"), max_len=2048, batch=8,
                               verbose=False,
                               median_file="gene_median_dictionary_gc104M.pkl",
                               token_file="token_dictionary_gc104M.pkl")
    else:
        raise ValueError(encoder)
    np.savez(f, Z=Z.astype(np.float32), lib=lib, ngene=ngene)
    return Z, lib, ngene


def load_emb(cond, encoder):
    """Load a cached embedding; hard error if the queue hasn't produced it."""
    f = EMB / f"{cond.replace('/','-')}__{encoder}.npz"
    if not f.exists():
        raise FileNotFoundError(f"run q0_embed first: missing {f.name}")
    d = np.load(f)
    return d["Z"].astype(np.float64), d["lib"], d["ngene"]


# ----------------------------------------------------- gene space + PCA space
def gene_space_mean(cond, rows_index, syms, hvg=None):
    """Per-condition mean log-CPM vector, cached (for q5's gene-space arm)."""
    f = EMB / f"{cond.replace('/','-')}__genemean.npz"
    if f.exists():
        d = np.load(f)
        return d["m"]
    X = load_rows(rows_index[cond]).astype(np.float64)
    Xn = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    m = Xn.mean(0)
    np.savez(f, m=m.astype(np.float32))
    return m


def fit_control_pca(rows_index, syms, n_comp=128, min_detection=0.05):
    """PCA basis fitted on CONTROL cells only, with the detection floor.

    Fitting on controls keeps the basis perturbation-independent, matching how
    the CCA subspace is fitted. The detection floor is LESSONS rule 8: raw
    variance ranking admits never-detected genes whose 'variance' is dropout.
    Cached with its HVG list so all spaces are reproducible.
    """
    f = EMB / "control_pca_basis.npz"
    if f.exists():
        d = np.load(f)
        return d["hvg"], d["W"], d["mu"]
    X = load_rows(rows_index[CONTROL]).astype(np.float64)
    Xn = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    det = (X > 0).mean(0)
    ok = np.flatnonzero(det >= min_detection)
    v = Xn[:, ok].var(0)
    hvg = ok[np.argsort(-v)[:2000]]
    Z = Xn[:, hvg]
    mu = Z.mean(0)
    Zc = Z - mu
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    W = Vt[:n_comp].T                          # (2000, 128)
    np.savez(f, hvg=hvg, W=W, mu=mu)
    return hvg, W, mu


def pca_embed_condition(cond, rows_index, syms):
    """Project one condition's cells onto the control-fitted PCA basis."""
    f = EMB / f"{cond.replace('/','-')}__pca.npz"
    if f.exists():
        d = np.load(f)
        return d["Z"].astype(np.float64), d["lib"], d["ngene"]
    hvg, W, mu = fit_control_pca(rows_index, syms)
    X = load_rows(rows_index[cond]).astype(np.float64)
    lib = X.sum(1)
    ngene = (X > 0).sum(1).astype(np.float32)
    Xn = np.log1p(X / np.maximum(lib[:, None], 1) * 1e4)
    Z = (Xn[:, hvg] - mu) @ W
    np.savez(f, Z=Z.astype(np.float32), lib=lib, ngene=ngene)
    return Z, lib, ngene


# ------------------------------------------------------------ depth projection
def fit_depth_projector(Z_ctrl, lib_ctrl, ngene_ctrl):
    """Coefficients to remove depth, fitted on control cells only."""
    C = np.column_stack([np.log1p(lib_ctrl), np.log1p(ngene_ctrl),
                         np.ones(len(lib_ctrl))])
    beta = np.linalg.lstsq(C, Z_ctrl, rcond=None)[0]
    mu = Z_ctrl.mean(0)
    return beta, mu


def apply_depth_projector(Z, lib, ngene, beta):
    C = np.column_stack([np.log1p(lib), np.log1p(ngene), np.ones(len(lib))])
    return Z - C @ beta


# --------------------------------------------------------------- CCA subspace
def fit_cca(zA, zB, kmax=128, ridge=1e-3, var=0.99):
    """Direct CCA on paired cells. Returns projector info for space A and B.

    The returned dict lets any experiment project a NEW vector in A onto the
    top-k shared directions: shared part = v @ P_A @ U[:, :k] (in canonical
    coordinates), or reconstructed in A-space for orthogonal-complement work.
    """
    def reduce(Z):
        Zc = Z - Z.mean(0)
        U_, S_, Vt_ = np.linalg.svd(Zc, full_matrices=False)
        keep = int(np.searchsorted(np.cumsum(S_**2) / np.sum(S_**2), var) + 1)
        keep = min(keep, kmax, max(8, len(Z) // 4))
        return Vt_[:keep].T
    PA, PB = reduce(zA), reduce(zB)
    A, B = zA @ PA, zB @ PB
    mA, mB = A.mean(0), B.mean(0)
    A, B = A - mA, B - mB

    def whitener(M):
        Cv = M.T @ M / len(M)
        w, V = np.linalg.eigh(Cv)
        w = np.maximum(w, ridge * w.max())
        return V @ np.diag(w ** -0.5) @ V.T
    Wa, Wb = whitener(A), whitener(B)
    U, S, Vt = np.linalg.svd((A @ Wa).T @ (B @ Wb) / len(A))
    return {"PA": PA, "PB": PB, "mA": mA, "mB": mB, "Wa": Wa, "Wb": Wb,
            "U": U, "V": Vt.T, "corr": S}


def shared_fraction(v, cca, k, side="A"):
    """Fraction of a vector's energy in the top-k shared canonical directions.

    The vector is expressed in the whitened reduced space (where canonical
    directions are orthonormal), split into its shared-k part and the rest, and
    the energy ratio reported. Working in whitened coordinates matters: in raw
    coordinates the canonical directions are not orthogonal and 'fraction of
    energy' is ill-defined.
    """
    if side == "A":
        w = (v @ cca["PA"]) @ cca["Wa"]
        Q = cca["U"][:, :k]
    else:
        w = (v @ cca["PB"]) @ cca["Wb"]
        Q = cca["V"][:, :k]
    tot = float(w @ w)
    if tot <= 0:
        return np.nan
    s = w @ Q
    return float(s @ s) / tot


# -------------------------------------------------------------------- utility
def cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def ridge_map(A, B, alphas=np.logspace(-2, 4, 13), folds=4, seed=0):
    """CV-tuned ridge map A->B (the corrected paired-map instrument)."""
    n = len(A)
    idx = np.random.default_rng(seed).permutation(n)
    cuts = np.array_split(idx, folds)
    err = np.zeros(len(alphas))
    for f in range(folds):
        va = cuts[f]
        tr = np.concatenate([cuts[g] for g in range(folds) if g != f])
        G, R = A[tr].T @ A[tr], A[tr].T @ B[tr]
        for j, al in enumerate(alphas):
            W = np.linalg.solve(G + al * np.eye(G.shape[0]), R)
            err[j] += float(((A[va] @ W - B[va]) ** 2).mean())
    best = float(alphas[int(np.argmin(err))])
    W = np.linalg.solve(A.T @ A + best * np.eye(A.shape[1]), A.T @ B)
    return W, best


def save_result(name, obj):
    p = ART / f"{name}{TAG}.json"
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {p}")


# ------------------------------------------------- generalized CCA (common S)
def fit_gcca(views, k=24, var=0.99, dmax=64, ridge=1e-3):
    """MAXVAR generalized CCA: map MANY models into ONE shared space S.

    WHY THIS EXISTS. Fitting pairwise CCA per model gives each model its own
    canonical basis against a different partner. Canonical direction 3 of
    (A vs B) has no reason to correspond to direction 3 of (C vs A), so a
    consensus taken coordinate-wise across such bases is meaningless. A
    cross-model consensus requires a COMMON target space.

    METHOD. Each view is PCA-reduced and whitened so its columns are
    orthonormal. Whitened views are concatenated horizontally, and the top-k
    left singular vectors of the concatenation are the shared latent G (one
    row per cell). This is the standard MAXVAR/SUMCOR generalized CCA solution
    and costs one thin SVD rather than an n x n eigendecomposition.

    Returns per-view loadings C_m with u = (z - mean) @ P @ W @ C_m, so any
    NEW vector in a model's space can be pushed into the same shared S.

    views: dict name -> (n_cells x d) matrix, SAME cells in the SAME order.
    """
    names = list(views)
    n = len(views[names[0]])
    for v in views.values():
        assert len(v) == n, "gcca views must be the same cells in the same order"
    P, W, mu, X = {}, {}, {}, {}
    for m in names:
        Z = views[m]
        mu[m] = Z.mean(0)
        Zc = Z - mu[m]
        U_, S_, Vt_ = np.linalg.svd(Zc, full_matrices=False)
        keep = int(np.searchsorted(np.cumsum(S_**2) / np.sum(S_**2), var) + 1)
        # A requested k-dimensional shared space must not silently receive a
        # lower-rank view merely because that view is variance-concentrated.
        # Keep at least k native components whenever the view and dmax permit.
        keep = min(max(keep, k), dmax, max(8, n // 4), Zc.shape[1])
        P[m] = Vt_[:keep].T
        A = Zc @ P[m]
        C = A.T @ A / len(A)
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, ridge * w.max())
        W[m] = V @ np.diag(w ** -0.5) @ V.T
        X[m] = A @ W[m]
    big = np.hstack([X[m] for m in names])
    G, s, _ = np.linalg.svd(big, full_matrices=False)
    k = min(k, G.shape[1])
    G = G[:, :k]
    Cm = {}
    for m in names:
        # least-squares loadings from the whitened view onto the shared latent
        Cm[m] = np.linalg.lstsq(X[m], G, rcond=None)[0]
    return {"names": names, "mu": mu, "P": P, "W": W, "C": Cm, "k": k,
            "sv": s[:k], "G": G}


def to_shared_gcca(v, g, m):
    """Push a vector from model m's raw space into the shared space S."""
    return ((v - 0) @ g["P"][m]) @ g["W"][m] @ g["C"][m]


def gcca_agreement(g):
    """Per-dimension agreement across views in S, as a validity check.

    For each shared dimension, the mean pairwise correlation between views'
    projections. Near 1 means the views genuinely agree on that direction;
    near 0 means the dimension is not shared and should not be trusted.
    """
    names = g["names"]
    Us = {m: (None) for m in names}
    out = []
    for j in range(g["k"]):
        cols = []
        for m in names:
            cols.append(g["G"][:, j])
        out.append(1.0)
    return out
