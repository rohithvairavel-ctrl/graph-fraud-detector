# Results

Held-out evaluation on synthetic payment graph with planted fraud rings.

## Primary comparison

| Model | ROC-AUC | PR-AUC | P@50 | P@100 | P@1% |
|-------|---------|--------|------|-------|------|
| Tabular only | 0.9069 | 0.2995 | 0.3 | 0.25 | 0.33 |
| Tabular + Graph | 0.987 | 0.9183 | 0.62 | 0.33 | 1.0 |
| RF + Graph | 0.9853 | 0.927 | 0.62 | 0.33 | 1.0 |
| Label Propagation | 0.5512 | 0.0628 | — | — | — |

## Lift

- ROC-AUC: +0.0801
- PR-AUC: +0.6188
- Precision@50: +0.32

## Setup

- Train n=1875, test n=625, fraud rate (test)=0.0544
- Modeling: sklearn on engineered graph features (degree, PageRank, Louvain, shared device/IP, cycles/triangles). GraphSAGE/torch-geometric omitted for clean install and reproducibility; feature-based approach is production-friendly and shows clear lift.
