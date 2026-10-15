"""Cell embedding extraction from scGPT and Geneformer.

Two genuinely different encoders over the same cells:

  scGPT       12-layer post-norm transformer, d=512, fused QKV. Input is a set of
              expressed genes plus their expression BINNED into 51 levels; the
              cell embedding is the <cls> output.
  Geneformer  6-layer BERT, d=256. Input is a RANK-VALUE encoding: genes ordered
              by expression normalised to each gene's corpus median, truncated to
              2048; the cell embedding is the mean of the final hidden states.

They share no tokenizer, no dimensionality, and no input representation, which is
what makes translation between them a real test rather than a re-parameterisation.
"""

from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# =====================================================================  scGPT
class ScGPT(nn.Module):
    """Faithful re-implementation of the scGPT encoder stack for inference."""

    def __init__(self, vocab_size=60697, d=512, nhead=8, nlayers=12, d_ff=512):
        super().__init__()
        self.d, self.nhead = d, nhead
        self.gene_emb = nn.Embedding(vocab_size, d)
        self.enc_norm = nn.LayerNorm(d)
        self.val_l1 = nn.Linear(1, d)
        self.val_l2 = nn.Linear(d, d)
        self.val_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(d, 3 * d),
                "out": nn.Linear(d, d),
                "l1": nn.Linear(d, d_ff),
                "l2": nn.Linear(d_ff, d),
                "n1": nn.LayerNorm(d),
                "n2": nn.LayerNorm(d),
            }) for _ in range(nlayers)])

    def load(self, sd):
        g = lambda k: sd[k]
        self.gene_emb.weight.data = g("encoder.embedding.weight")
        self.enc_norm.weight.data = g("encoder.enc_norm.weight")
        self.enc_norm.bias.data = g("encoder.enc_norm.bias")
        self.val_l1.weight.data = g("value_encoder.linear1.weight")
        self.val_l1.bias.data = g("value_encoder.linear1.bias")
        self.val_l2.weight.data = g("value_encoder.linear2.weight")
        self.val_l2.bias.data = g("value_encoder.linear2.bias")
        self.val_norm.weight.data = g("value_encoder.norm.weight")
        self.val_norm.bias.data = g("value_encoder.norm.bias")
        for i, L in enumerate(self.layers):
            p = f"transformer_encoder.layers.{i}."
            L["qkv"].weight.data = g(p + "self_attn.Wqkv.weight")
            L["qkv"].bias.data = g(p + "self_attn.Wqkv.bias")
            L["out"].weight.data = g(p + "self_attn.out_proj.weight")
            L["out"].bias.data = g(p + "self_attn.out_proj.bias")
            L["l1"].weight.data = g(p + "linear1.weight")
            L["l1"].bias.data = g(p + "linear1.bias")
            L["l2"].weight.data = g(p + "linear2.weight")
            L["l2"].bias.data = g(p + "linear2.bias")
            L["n1"].weight.data = g(p + "norm1.weight")
            L["n1"].bias.data = g(p + "norm1.bias")
            L["n2"].weight.data = g(p + "norm2.weight")
            L["n2"].bias.data = g(p + "norm2.bias")
        return self

    def forward(self, ids, vals, pad_mask):
        h = self.enc_norm(self.gene_emb(ids))
        v = self.val_norm(self.val_l2(torch.relu(self.val_l1(vals.unsqueeze(-1)))))
        h = h + v
        B, S, D = h.shape
        for L in self.layers:
            qkv = L["qkv"](h).view(B, S, 3, self.nhead, D // self.nhead)
            q, k, v_ = qkv.permute(2, 0, 3, 1, 4)
            # SDPA boolean attn_mask semantics: True = ATTEND. pad_mask is built
            # True for real tokens, so it is passed directly. Inverting it here
            # made every cell attend only to padding, which collapsed the whole
            # embedding to rank 1 (participation ratio 1.0, cell-type probe at
            # the majority-class rate). The sanity probe caught it.
            att = torch.nn.functional.scaled_dot_product_attention(
                q, k, v_, attn_mask=pad_mask[:, None, None, :])
            a = L["out"](att.transpose(1, 2).reshape(B, S, D))
            h = L["n1"](h + a)                                   # post-norm
            f = L["l2"](torch.relu(L["l1"](h)))
            h = L["n2"](h + f)
        return h[:, 0]                                            # <cls>


def bin_expression(x, n_bins=51):
    """scGPT value binning: non-zero expression -> quantile bins, per cell."""
    out = np.zeros_like(x, dtype=np.float32)
    nz = x > 0
    if nz.sum() == 0:
        return out
    v = x[nz]
    edges = np.quantile(v, np.linspace(0, 1, n_bins))
    out[nz] = np.clip(np.digitize(v, edges[1:-1]) + 1, 1, n_bins - 1)
    return out


def embed_scgpt(X, gene_names, model_dir, max_len=1200, batch=16, verbose=True):
    """X: cells x genes raw counts. Returns n x 512 cell embeddings."""
    md = pathlib.Path(model_dir)
    vocab = json.load(open(md / "vocab.json"))
    sd = torch.load(md / "best_model.pt", map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    dev = device()
    m = ScGPT().load(sd).to(dev).eval()

    keep = [(i, vocab[g]) for i, g in enumerate(gene_names) if g in vocab]
    if verbose:
        print(f"  scGPT: {len(keep)}/{len(gene_names)} genes in vocab", flush=True)
    cols = np.array([i for i, _ in keep])
    toks = np.array([t for _, t in keep])
    cls_id, pad_id = vocab["<cls>"], vocab["<pad>"]

    Xs = np.asarray(X)[:, cols]
    Xs = np.log1p(Xs / np.maximum(Xs.sum(1, keepdims=True), 1) * 1e4)

    out = []
    for s in range(0, len(Xs), batch):
        chunk = Xs[s:s + batch]
        ids_b, val_b, msk_b = [], [], []
        for row in chunk:
            nz = np.flatnonzero(row)
            if len(nz) > max_len - 1:
                nz = nz[np.argsort(row[nz])[::-1][:max_len - 1]]
            b = bin_expression(row[nz])
            ids = np.concatenate([[cls_id], toks[nz]])
            vals = np.concatenate([[0.0], b])
            pad = max_len - len(ids)
            ids_b.append(np.concatenate([ids, np.full(pad, pad_id)]))
            val_b.append(np.concatenate([vals, np.zeros(pad)]))
            msk_b.append(np.concatenate([np.ones(len(ids), bool), np.zeros(pad, bool)]))
        with torch.no_grad():
            e = m(torch.tensor(np.array(ids_b), dtype=torch.long, device=dev),
                  torch.tensor(np.array(val_b), dtype=torch.float32, device=dev),
                  torch.tensor(np.array(msk_b), device=dev))
        out.append(e.cpu().numpy())
        if verbose and s % (batch * 20) == 0:
            print(f"    {s}/{len(Xs)}", flush=True)
    return np.concatenate(out).astype(np.float32)


# ================================================================  Geneformer
def embed_geneformer(X, ensembl_ids, model_dir, max_len=2048, batch=16, verbose=True,
                     median_file="gene_median_dictionary_gc30M.pkl",
                     token_file="token_dictionary_gc30M.pkl"):
    """Rank-value encoding then mean-pooled BERT hidden states.

    The dictionary filenames differ between Geneformer versions (gc30M for V1,
    gc104M for V2) and the vocabularies are NOT interchangeable, so they are
    parameters rather than constants.
    """
    from transformers import BertModel
    md = pathlib.Path(model_dir)
    gmed = pickle.load(open(md / median_file, "rb"))
    tok = pickle.load(open(md / token_file, "rb"))
    dev = device()
    m = BertModel.from_pretrained(str(md)).to(dev).eval()

    keep = [(i, tok[e], gmed[e]) for i, e in enumerate(ensembl_ids)
            if e in tok and e in gmed and gmed[e] > 0]
    if verbose:
        print(f"  Geneformer: {len(keep)}/{len(ensembl_ids)} genes tokenised", flush=True)
    cols = np.array([i for i, _, _ in keep])
    toks = np.array([t for _, t, _ in keep])
    meds = np.array([g for _, _, g in keep], dtype=np.float32)

    Xs = np.asarray(X)[:, cols].astype(np.float32)
    Xs = Xs / np.maximum(Xs.sum(1, keepdims=True), 1) * 1e4
    Xs = Xs / meds[None, :]                                    # median-normalise

    pad = tok.get("<pad>", 0)
    out = []
    for s in range(0, len(Xs), batch):
        chunk = Xs[s:s + batch]
        ids_b, msk_b = [], []
        for row in chunk:
            nz = np.flatnonzero(row)
            order = nz[np.argsort(row[nz])[::-1]][:max_len]     # rank-value
            ids = toks[order]
            p = max_len - len(ids)
            ids_b.append(np.concatenate([ids, np.full(p, pad)]))
            msk_b.append(np.concatenate([np.ones(len(ids), int), np.zeros(p, int)]))
        ids_t = torch.tensor(np.array(ids_b), dtype=torch.long, device=dev)
        msk_t = torch.tensor(np.array(msk_b), dtype=torch.long, device=dev)
        with torch.no_grad():
            h = m(input_ids=ids_t, attention_mask=msk_t).last_hidden_state
            w = msk_t.unsqueeze(-1).float()
            e = (h * w).sum(1) / w.sum(1).clamp(min=1)          # mean pool
        out.append(e.cpu().numpy())
        if verbose and s % (batch * 20) == 0:
            print(f"    {s}/{len(Xs)}", flush=True)
    return np.concatenate(out).astype(np.float32)
