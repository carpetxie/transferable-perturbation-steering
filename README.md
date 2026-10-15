# Transferable perturbation steering across single-cell foundation models

## Question

Single-cell foundation models (scGPT, Geneformer) embed the same cell in
unrelated coordinate systems. If a perturbation's effect is measured in one
model, can it be carried into another model that has no perturbation labels,
and used to steer predicted gene expression?

## Method

1. Embed the same cells with several frozen encoders: scGPT, Geneformer V1
   (10M), V2 (104M), V2 (316M), plus scVI and expression PCA.
2. Regress sequencing depth (log library size, log genes detected) out of every
   space, fit on control cells only.
3. Fit one shared space with generalized CCA (MAXVAR, K = 24) on **control cells
   only**, so no perturbation outcome enters the basis.
4. Estimate each perturbation's effect in donor models, map it through the
   shared space into a held-out target model, and score it against the target's
   own measured effect (leave-one-model-out).
5. Split every condition by **cell**, identically across all encoders, so donor
   and gold estimates never share cells. Report every score against the
   target's split-half noise ceiling.

Data: Norman2019 K562 Perturb-seq (201 perturbations, 6000 controls), with
Adamson2016 and Papalexi2021 as replication datasets.

## Results

### 1. Zero-label transfer reaches 94-97% of the noise ceiling

| arm (Norman, leave-one-model-out) | centroid cosine |
|---|---:|
| transferred from the other models | **0.531** |
| fit natively in the target model | 0.565 |
| split-half noise ceiling | 0.564 |
| shuffled perturbation labels | 0.240 |
| matched random basis | -0.002 |

With a fixed shared space across four encoders, transfer reaches 94-97% of
each target's ceiling, and donor choice barely matters (all donors within 0.005
of the best subset). The random basis transfers nothing, so the learned shared
space carries the effect.

### 2. The transfer is perturbation-specific

Ranking the transferred effect against all 201 perturbations: top-1 = 0.919,
mean rank 1.4 / 201 after removing the generic treated-vs-control axis
(chance 0.005).

### 3. One labelled pilot perturbation beats native estimation

Donor-to-target error is mostly a systematic bias shared across perturbations.
Estimating it from a single labelled pilot perturbation and subtracting it
(cross-fitted, cell-disjoint):

| dataset | native | debiased transfer | gain |
|---|---:|---:|---|
| Norman | 0.532 | **0.732** | +0.200 [+0.186, +0.214] |
| Adamson | 0.365 | **0.772** | +0.408 [+0.380, +0.434] |
| Papalexi | 0.524 | **0.840** | +0.316 [+0.294, +0.337] |

### 4. Any faithful cell representation can join the shared space

Expression PCA and scVI exchange perturbation directions with the transformer
models. Transformers transfer into scVI at 87% and PCA at 92% of their ceilings;
scVI/PCA donors transfer back at 94-98%. Across all six models, transfer
averages 95% of ceiling.

### 5. Transferred effects steer a nonlinear gene-expression model

Two MLP decoders (latent to gene expression) are trained on control cells only.
Each transferred effect is added to 256 held-out control cells and decoded.
Using a donor rule frozen before the target's outcomes are seen:

| | MLP 1 | MLP 2 |
|---|---:|---:|
| gene-level DE cosine | 0.578 | 0.576 |
| wrong-perturbation floor | 0.412 | 0.410 |
| advantage, generic direction removed | +0.190 [0.157, 0.226] | +0.189 [0.156, 0.223] |
| real perturbed cells (ceiling) | 0.648 | 0.646 |

The decoder never saw a perturbed cell, and the response depends on the
starting cell state (non-additivity ratio 0.41-0.45). The same pipeline is
significant in all 9 (dataset, target) combinations.

### 6. Practical value

With no labels in a new encoder, a transferred effect is worth about 20
labelled cells per perturbation there. The linear map holds far from controls:
fidelity rises from 0.551 to 0.720 with distance, and linear beats MLP in every
bin.

## Scope

- Steering here is of a decoder trained on embeddings, not of a foundation
  model's internal computation, and not of real cells.
- What transfers is the main effect as an additive vector. Interaction
  (epistatic) residuals are not carried above sampling noise.
- Without a pilot, transfer is slightly below native estimation at equal cell
  budget (-0.03 to -0.08). The pilot debias reverses that.
- Encoders read the same physical cells, so this is transfer across coordinate
  systems, not across donors or studies.

## Layout

```
code/embedding/     embed cells with each encoder (Modal GPU jobs)
code/shared_space/  paired maps, CCA, nulls, depth removal, gene sets
code/pipeline/      qlib.py (shared GCCA library) + one script per result
results/            JSON output for every number above
tests/              GCCA and baseline tests
```

Result to script:

| result | script | output |
|---|---|---|
| 1 | `n2_equivariance.py`, `e33_donor_ranking.py` | `n2_equivariance.json`, `e33_donor_ranking.json` |
| 2 | `n2b_specificity.py` | `n2b_specificity.json` |
| 3 | `e34_debias.py` .. `e37_debias_leakfix.py` | `e37_debias_leakfix_*.json` |
| 4 | `e38_pca_member.py`, `e39_architecture_diversity.py`, `e40_modal_control_alignment.py` | matching JSON |
| 5 | `e40_nonlinear_steering.py`, `e41_debias_steering.py`, `e42_nested_zero_label_selection.py`, `e43_*` | `e43_nonlinear_steering.json` |
| 6 | `e3_compiler.py`, `q4_trust_radius.py` | matching JSON |

Model weights, `.h5ad` data and cached embeddings are not included. Scripts
expect them under `~/.cache/silico_vec2vec/`.
