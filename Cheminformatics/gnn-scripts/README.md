# GNN Architecture Comparison — Cheminformatics & Drug Discovery

**Author: Himanshu Goel** | [Website](https://himanshugoel.github.io) | [GitHub](https://github.com/himanshugoel)

Five Python scripts covering the complete progression of Graph Neural Network architectures applied to molecular property prediction — from basic GCN to state-of-the-art Graph Transformers, with a comprehensive benchmark comparison.

---

## Scripts

| Script | Architecture | Task | Key Innovation |
|--------|-------------|------|---------------|
| `01_gcn_baseline.py` | **GCN** (Kipf 2017) | hERG cardiotoxicity | Topology only, symmetric normalization |
| `02_mpnn_edge_features.py` | **MPNN** (Gilmer 2017) | DILI hepatotoxicity | Edge features + GRU update + Set2Set |
| `03_gat_attention.py` | **GAT** (Veličković 2018) | Solubility (logS) | Multi-head attention + JK connections |
| `04_graph_transformer.py` | **GraphTransformer** + extensions | Tox21 (12 endpoints) | Global attention + virtual node + 3D + LPE |
| `05_benchmark_comparison.py` | **GCN vs MPNN vs GAT vs GIN vs GraphTransf** | hERG (unified) | Full Pareto/radar/stability analysis |

---

## Architecture progression

```
GCN (2017)
  → ignores edge features
  → symmetric adjacency normalization

MPNN (2017)
  + edge features via NNConv (message MLP)
  + GRU-based node update (stateful)
  + Set2Set readout (attention-based pooling)

GAT / GATv2 (2018/2022)
  + learnable attention weights α_{ij}
  + multi-head attention (8 heads)
  + interpretable — which neighbors matter?
  + Jumping Knowledge connections

GIN (2019)
  + maximally expressive (= 1-WL test)
  + learnable epsilon parameter
  + theoretically distinguishes more graphs than GCN/GAT

Graph Transformer (2021+)
  + global self-attention over all atoms
  + virtual node (super-node → global memory)
  + 3D conformer distances (RBF encoded)
  + Laplacian positional encoding
  + uncertainty-weighted multi-task loss (Kendall 2018)
```

---

## Installation

```bash
# Core dependencies
pip install torch torch-geometric rdkit scikit-learn numpy pandas matplotlib

# For PyG (follow official guide for your CUDA version)
pip install torch-geometric

# Full environment
pip install -r requirements.txt
```

---

## Running

```bash
# Run all scripts sequentially
python 01_gcn_baseline.py
python 02_mpnn_edge_features.py
python 03_gat_attention.py
python 04_graph_transformer.py
python 05_benchmark_comparison.py   # loads results from 01-04 automatically
```

Results and plots saved to `gnn_results/`.

---

## Task coverage

| Toxicity endpoint | Script | Regulatory framework |
|---|---|---|
| hERG cardiotoxicity | 01, 05 | ICH E14/S7B, CiPA |
| DILI hepatotoxicity | 02 | ICH S2, FDA DILIrank |
| Solubility (logS) | 03 | BCS classification |
| Tox21 (12 endpoints) | 04 | ICH S7, FDA Tox21 |

---

## Key design principles

- **Scaffold splitting** used throughout (more realistic than random)
- **Class imbalance** handled via pos_weight in BCEWithLogitsLoss
- **Multi-seed evaluation** (Script 05) for statistical robustness
- **Pareto analysis**: performance vs model complexity
- **Interpretability**: attention weights extractable from GAT/GraphTransf
- **Uncertainty quantification**: MC Dropout (Script 03), homoscedastic (Script 04)

---

## Scientific references

1. Kipf & Welling (2017) — Semi-Supervised Classification with GCNs. ICLR.
2. Gilmer et al. (2017) — Neural Message Passing for Quantum Chemistry. ICML.
3. Veličković et al. (2018) — Graph Attention Networks. ICLR.
4. Xu et al. (2019) — How Powerful are GNNs? ICLR.
5. Shi et al. (2021) — Masked Label Prediction + Graph Transformer. arXiv.
6. Brody et al. (2022) — How Attentive are GATs? (GATv2). ICLR.
7. Ying et al. (2021) — Graphormer (virtual node, spatial encoding). NeurIPS.
8. Kendall & Gal (2018) — Multi-task learning using uncertainty. CVPR.

---

## Connection to my research

These scripts build on computational methods from my published work:
- hERG prediction model (*Chemistry* 2022, 4, 630–646)
- DILI/hepatotoxicity (*Int. J. Mol. Sci.* 2024, 25, 3265)
- Kidney injury toxicogenomics (*Int. J. Mol. Sci.* 2023, 24, 7434)
- MEA neurotoxicity (BHSAI/HJF, 2022-2025)
