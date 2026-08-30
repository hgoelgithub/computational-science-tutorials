# Generative AI for Drug Discovery & Cheminformatics

**Author: Himanshu Goel** | [Website](https://hgoelgithub.github.io)

Five Python scripts covering the complete progression of **Generative AI architectures** for de novo drug design and molecular generation — from basic SMILES-RNN to state-of-the-art Diffusion models, with a comprehensive benchmark comparison.

---

## Architecture Progression

| Script | Architecture | Paper | Key Innovation |
|--------|-------------|-------|---------------|
| `01_smiles_rnn.py` | **SMILES RNN** (LSTM) | Segler et al. 2018 ACS Cent. Sci. | Char-level language model → REINVENT 1.0 |
| `02_cvae_molecular.py` | **Conditional VAE** | Gómez-Bombarelli 2018 ACS Cent. Sci. | Continuous latent + property conditioning |
| `03_04_05_...py` | **MolGAN** | De Cao & Kipf 2018 (ICML workshop) | Adversarial training + property reward |
| (same file) | **Transformer + RL** | REINVENT 4, Loeffler 2024 J. Cheminform. | GPT Transformer + REINFORCE oracle |
| (same file) | **DDPM Diffusion** | Austin 2021 (D3PM), Vignac 2022 (DiGress) | Non-autoregressive denoising from noise |

---

## Benchmark Metrics (MOSES standard)

| Metric | Definition |
|--------|-----------|
| **Validity** | % of generated SMILES that parse to valid RDKit molecules |
| **Uniqueness** | % of valid molecules that are unique (deduplicated) |
| **Novelty** | % of unique valid molecules NOT in training set |
| **Diversity** | Mean pairwise Tanimoto distance (higher = more diverse) |
| **Drug-likeness** | % satisfying Lipinski Ro5 (MW≤500, LogP≤5, HBD≤5, HBA≤10) |
| **QED** | Quantitative Estimate of Drug-likeness (Bickerton 2012, 0-1) |
| **SA score** | Synthetic Accessibility (Ertl & Schuffenhauer 2009, 1-10) |

---

## Literature Performance Summary

| Model | Validity | Novelty | Diversity | QED | Key strength |
|-------|---------|---------|-----------|-----|-------------|
| SMILES RNN | 0.86 | 0.78 | 0.83 | 0.56 | Fast, simple baseline |
| CVAE | 0.75 | 0.91 | 0.88 | 0.60 | Property control, latent interpolation |
| MolGAN | 0.68 | 0.88 | 0.71 | 0.58 | Reward shaping (mode collapse risk) |
| Transformer+RL | **0.94** | 0.85 | 0.87 | **0.70** | Production standard (REINVENT 4) |
| DDPM Diffusion | 0.71 | **0.94** | **0.91** | 0.63 | Max diversity, 3D-capable |

---

## Run

```bash
pip install torch rdkit numpy pandas matplotlib scikit-learn

# Run individually
python 01_smiles_rnn.py
python 02_cvae_molecular.py
python 03_04_05_gan_transformer_diffusion.py  # Runs 03+04+05 + full comparison
```

Results and plots saved to `genai_results/`.

---

## Key design details

### Script 01 — SMILES RNN
- 3-layer LSTM, 512 hidden units
- Temperature-controlled sampling (T=0.9 default)
- Teacher forcing training
- This architecture underpins REINVENT 1.0-3.0 (AstraZeneca)

### Script 02 — CVAE
- Bidirectional GRU encoder → (μ, σ²) → 64-dim latent
- Property conditioning: [MW, LogP, TPSA, QED] concatenated to z
- KL annealing (β: 0→1 over 30 epochs) prevents posterior collapse
- Free bits (min KL per dimension) further prevents collapse
- Latent space interpolation: smooth chemical pathways between molecules

### Script 03 — MolGAN
- LSTM Generator + BiLSTM Discriminator
- ORGAN-style property reward = 0.5×Ro5 + 0.5×QED
- Wasserstein-style adversarial training
- n_disc_steps=2 (D trained twice per G step)

### Script 04 — Transformer + RL (REINVENT 4-style)
- GPT-style autoregressive Transformer (4 layers, 4 heads)
- Two-phase training: Prior (distribution learning) → RL fine-tuning
- REINFORCE with baseline: reward = QED + drug-likeness − hERG_penalty − SA_penalty
- Augmented Hill-Climb (AHC) variant used in REINVENT 4
- This is the production architecture at AstraZeneca, Novartis, Pfizer

### Script 05 — DDPM Diffusion
- Absorbing-state discrete diffusion (D3PM, Austin 2021)
- Transformer denoiser with sinusoidal time embeddings
- Cosine noise schedule: 50 forward/reverse steps
- Non-autoregressive — all tokens generated in parallel
- Pathway to 3D molecular graph diffusion (DiGress, DiffSBDD, TargetDiff)

---

## References

1. Segler et al. 2018 — Generating Focused Molecule Libraries. ACS Cent. Sci.
2. Gómez-Bombarelli et al. 2018 — Automatic Chemical Design using VAE. ACS Cent. Sci.
3. De Cao & Kipf 2018 — MolGAN: An implicit generative model for small molecules.
4. Loeffler et al. 2024 — REINVENT 4. J. Cheminform. (AstraZeneca).
5. Austin et al. 2021 — Structured Denoising Diffusion (D3PM). NeurIPS.
6. Vignac et al. 2022 — DiGress: Discrete Diffusion for graphs. ICLR 2023.
7. Polykovskiy et al. 2020 — MOSES: A Molecular Generation Benchmark. Front. Pharmacol.
8. Polykovskiy et al. 2018 — GuacaMol benchmark. J. Chem. Inf. Model.

---

## Connection to my research

- hERG prediction → reward function in Script 04 RL
- DILI/hepatotoxicity → multi-objective oracle design
- Toxicogenomics work (BHSAI) → safety-aware molecular generation
- MEA neurotoxicity → property penalty terms
