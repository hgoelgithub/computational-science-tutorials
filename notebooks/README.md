# Computational Science Tutorials

**By Himanshu Goel** · [Website](https://himanshugoel.github.io) 
A collection of Jupyter notebooks covering computational drug discovery, cheminformatics, bioinformatics, agentic AI, and machine learning — all runnable in Google Colab with one click.

---

## Drug Design & CADD

| # | Tutorial | Open in Colab |
|---|----------|---------------|
| 1 | Molecular fingerprints & Tanimoto similarity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/drug_design/01_fingerprints_similarity.ipynb) |
| 2 | Lipinski Ro5 & ADMET profiling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/drug_design/02_ro5_admet.ipynb) |
| 3 | Virtual screening with AutoDock Vina | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/drug_design/03_vina_screening.ipynb) |
| 4 | hERG cardiotoxicity prediction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/drug_design/04_herg_prediction.ipynb) |

## Cheminformatics

| # | Tutorial | Open in Colab |
|---|----------|---------------|
| 5 | RDKit from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/cheminformatics/05_rdkit_basics.ipynb) |
| 6 | Chemical space visualization with UMAP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/cheminformatics/06_chemical_space_umap.ipynb) |
| 7 | SMARTS substructure search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/cheminformatics/07_smarts_search.ipynb) |

## Bioinformatics & Toxicogenomics

| # | Tutorial | Open in Colab |
|---|----------|---------------|
| 8 | RNA-Seq QC & differential expression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/bioinformatics/08_rnaseq_deseq2.ipynb) |
| 9 | In vitro–in vivo correlation (IVIVE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/bioinformatics/09_ivive_correlation.ipynb) |
| 10 | MEA neurotoxicity & dose-response modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/bioinformatics/10_mea_dose_response.ipynb) |

## Agentic AI for Science

| # | Tutorial | Open in Colab |
|---|----------|---------------|
| 11 | RAG pipeline over scientific literature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/agentic_ai/11_rag_pubmed.ipynb) |
| 12 | LangGraph agent: ChEMBL + PubChem | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/agentic_ai/12_langgraph_chembl.ipynb) |

## Machine Learning & Deep Learning

| # | Tutorial | Open in Colab |
|---|----------|---------------|
| 13 | Classical ML baseline for QSAR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/ml_dl/13_qsar_classical_ml.ipynb) |
| 14 | Graph Neural Networks for molecules | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/ml_dl/14_gnn_molecules.ipynb) |
| 15 | Fine-tuning ChemBERTa for toxicity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/ml_dl/15_chemberta_finetune.ipynb) |

---

## Run locally

```bash
git clone https://github.com/himanshugoel/computational-science-tutorials
cd computational-science-tutorials
pip install -r requirements.txt
jupyter lab
```

## About

These tutorials are based on my research at BHSAI (Henry M. Jackson Foundation) and the University of Maryland Baltimore CADD Center. They cover the computational workflows I use daily for drug discovery, toxicology, and AI/ML in the life sciences.

If you find them useful, please ⭐ star the repo!
