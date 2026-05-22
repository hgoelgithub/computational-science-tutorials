# Computational Toxicology Notebooks

**Author: Himanshu Goel** | [Website](https://himanshugoel.github.io) | [GitHub](https://github.com/himanshugoel)

10 industry-standard Jupyter notebooks covering the full computational toxicology framework used in pharma drug safety and regulatory submissions.

## Notebooks

| # | Topic | Open in Colab | Regulatory Framework |
|---|-------|--------------|---------------------|
| 01 | Tox21 Multi-task Benchmark | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/01_tox21/01_tox21_multitask_benchmark.ipynb) | ICH S2/S7, FDA Tox21 |
| 02 | DILI / Hepatotoxicity | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/02_dili/02_dili_hepatotoxicity.ipynb) | ICH S2, FDA DILIrank |
| 03 | Cardiotoxicity (hERG + CiPA) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/03_cardiotox/03_cardiotox_herg_cipa.ipynb) | ICH E14/S7B, CiPA |
| 04 | Neurotoxicity & BBB | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/04_neurotox_bbb/04_neurotox_bbb.ipynb) | ICH S7A, OECD 424 |
| 05 | Nephrotoxicity (Kidney) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/05_nephrotox/05_nephrotoxicity_kidney.ipynb) | ICH S7A, FDA biomarkers |
| 06 | Genotoxicity (Ames/ICH M7) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/06_genotox/06_genotoxicity_mutagenicity.ipynb) | ICH M7(R2), OECD 471 |
| 07 | Acute LD50 / GHS | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/07_ld50/07_acute_oral_toxicity_ld50.ipynb) | GHS, OECD TG 423 |
| 08 | Multi-task Deep ToxNet | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/08_multitask_dnn/08_multitask_deeptoxnet.ipynb) | Industry DNN standard |
| 09 | Explainable AI (SHAP + AD) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/09_shap_explainability/09_explainable_ai_toxicology.ipynb) | OECD 5 principles, ICH M7 |
| 10 | Integrated Pipeline + Report | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshugoel/computational-science-tutorials/blob/main/toxicology/10_integrated_pipeline/10_integrated_tox_pipeline.ipynb) | IATA, NGRA framework |

## Run locally
```bash
git clone https://github.com/himanshugoel/computational-science-tutorials
cd computational-science-tutorials/toxicology
pip install -r requirements.txt
jupyter lab
```

## Scientific basis
These notebooks are grounded in my research at:
- **BHSAI / Henry M. Jackson Foundation** (2022-2025): MEA neurotoxicity, RNA-Seq toxicogenomics, ADMET profiling
- **UMB CADD Center** (2018-2022): hERG prediction (*Chemistry* 2022), SILCS drug design
- **Published work**: *Int. J. Mol. Sci.* 2023 (kidney injury), 2024 (liver injury), *Chem. Sci.* 2021
