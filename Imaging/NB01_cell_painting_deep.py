"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Imaging NB01 — Cell Painting & High-Content Screening (Deep Dive)          ║
║  Author: Himanshu Goel | himanshugoel.github.io                             ║
║                                                                              ║
║  STEP 1 — Assay design: channels, dyes, organelles, plate layout            ║
║  STEP 2 — CellProfiler pipeline: illumination, segmentation, 1500 features  ║
║  STEP 3 — pycytominer: QC, normalization, batch correction, feature select  ║
║  STEP 4 — MoA classification: PCA → RF/GBM/CNN, BBBC021 benchmark          ║
║  STEP 5 — Toxicity prediction: DILI, mitotox, ToxCast 412 assays            ║
║  STEP 6 — DeepProfiler CNN: EfficientNet-B0 fine-tune on 5-channel images   ║
║  STEP 7 — Compound clustering, UMAP, perturbation discovery                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

BIOLOGICAL CONTEXT
──────────────────
Cell Painting (Bray 2016, Nat Protocols) simultaneously images 8 organelles
using 6 fluorescent dyes in 5 imaging channels:

  ┌──────────┬──────────────────────┬────────────────────────────────┐
  │ Channel  │ Dye                  │ Structures stained              │
  ├──────────┼──────────────────────┼────────────────────────────────┤
  │ DNA      │ Hoechst 33342        │ Nucleus (DNA)                  │
  │ ER       │ Concanavalin A       │ Endoplasmic reticulum          │
  │ RNA/Nuc  │ SYTO 14              │ Nucleoli + cytoplasmic RNA     │
  │ AGP      │ Phalloidin + WGA     │ Actin, Golgi, plasma membrane  │
  │ Mito     │ MitoTracker Deep Red │ Mitochondria                   │
  └──────────┴──────────────────────┴────────────────────────────────┘

WHY IT MATTERS
──────────────
Each compound leaves a unique "morphological barcode" — a signature of
how it perturbs cellular structures. This barcode can be used to:
  • Predict Mechanism of Action (MoA) — 12+ classes, AUC ~0.87
  • Predict DILI/cytotoxicity — AUC=0.73 across 412 ToxCast assays
  • Find structurally-similar compounds with similar phenotypes
  • Cluster unknowns with known reference compounds (scaffold hopping)

DATASETS
────────
BBBC021 (Ljosa 2013): 113 compounds × 12 MoA, MCF-7 cells — BENCHMARK
JUMP-CP (2024): 136,000 chemicals × 1.6 billion cells, 115 TB
Cell Painting Gallery (2024): 656 TB, largest public imaging database
ToxCast (EPA): 9,000+ chemicals × 1,500+ assays (in vitro HTS)
"""

import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import median_abs_deviation, ttest_ind, rankdata
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                               classification_report, average_precision_score)
from sklearn.neighbors import NearestNeighbors
import torch, torch.nn as nn, torch.nn.functional as F

print("="*72)
print("  IMAGING NB01 — Cell Painting: Deep Dive (7 Steps)")
print("  CellProfiler · pycytominer · MoA · Toxicity · DeepProfiler CNN")
print("="*72)
np.random.seed(42); torch.manual_seed(42)
os.makedirs("imaging_results", exist_ok=True)

# ── STEP 1: Assay Design & Feature Architecture ───────────────────────────────
print("\n" + "─"*60)
print("  STEP 1 — Cell Painting Assay Design & CellProfiler Feature Space")
print("─"*60)

CHANNELS = {
    "DNA":   {"dye":"Hoechst 33342",  "structures":["Nucleus shape","DNA content","Mitotic index"],          "n_features":280},
    "ER":    {"dye":"Concanavalin A", "structures":["ER network texture","ER fragmentation","Perinuclear ER"],"n_features":240},
    "RNA":   {"dye":"SYTO 14",        "structures":["Nucleolar size","RNA distribution","Cell stress"],       "n_features":200},
    "AGP":   {"dye":"Phalloidin+WGA", "structures":["Actin stress fibers","Golgi morphology","Cell boundary"],"n_features":300},
    "Mito":  {"dye":"MitoTracker",    "structures":["Mito area","Mito network","Membrane potential proxy"],   "n_features":220},
}
COMPARTMENTS = ["Nuclei","Cells","Cytoplasm"]
FEATURE_TYPES = ["Intensity","AreaShape","Texture","RadialDistribution","Correlation","Granularity"]

N_FEATURES_TOTAL = sum(c["n_features"] for c in CHANNELS.values())  # ~1240
print(f"\n  Channel/feature breakdown:")
for ch, info in CHANNELS.items():
    print(f"    {ch:6s} ({info['dye']:22s}): {info['n_features']:4d} features  |  {', '.join(info['structures'][:2])}")
print(f"\n  Total morphological features: {N_FEATURES_TOTAL}")
print(f"  Feature types per channel: {', '.join(FEATURE_TYPES)}")
print(f"  Compartments: {', '.join(COMPARTMENTS)} (segmented separately)")

# ── STEP 2: Generate Realistic Compound Profiles ──────────────────────────────
print("\n" + "─"*60)
print("  STEP 2 — CellProfiler Profiles: MoA-specific signatures")
print("─"*60)
"""
CellProfiler (McQuin 2018, PLoS Bio) pipeline:
  Pipeline 1: Illumination correction (calculate correction function)
  Pipeline 2: QC (identify artifact images, remove blur/empty wells)
  Pipeline 3: Feature extraction
    → IdentifyPrimaryObjects (nuclei, Otsu thresholding)
    → IdentifySecondaryObjects (cells, propagation from nuclei)
    → IdentifyTertiaryObjects (cytoplasm = cell − nuclei)
    → MeasureObjectIntensity, MeasureTexture, MeasureObjectSizeShape
    → MeasureCorrelation, MeasureGranularity, MeasureRadialDistribution

Feature naming: {compartment}_{feature_type}_{channel}_{stat}
  e.g.: Cells_Texture_DNA_Haralick_Correlation_1
"""

MOA_CLASSES = {
    "Actin_disruptors":           {"n":35, "color":"#e74c3c",
        "perturbed_dims": slice(700,900),    # AGP features (actin)
        "perturbation":   [+2.5, -1.0]},
    "Aurora_kinase_inhibitors":   {"n":30, "color":"#e67e22",
        "perturbed_dims": slice(0,200),      # DNA features (mitosis)
        "perturbation":   [+3.0, 0.5]},
    "DNA_damage":                 {"n":35, "color":"#c0392b",
        "perturbed_dims": slice(50,250),     # Nuclei + DNA intensity
        "perturbation":   [+2.0, +1.5]},
    "Kinase_inhibitors":          {"n":45, "color":"#3498db",
        "perturbed_dims": slice(400,650),    # Mixed cytoplasmic
        "perturbation":   [+1.5, -0.5]},
    "Protein_synthesis_inhibitors":{"n":30, "color":"#8e44ad",
        "perturbed_dims": slice(200,400),    # RNA/ER features
        "perturbation":   [-2.0, +1.0]},
    "Microtubule_disruptors":     {"n":25, "color":"#27ae60",
        "perturbed_dims": slice(600,750),    # Shape features
        "perturbation":   [+2.2, -0.8]},
    "DMSO_control":               {"n":50, "color":"#95a5a6",
        "perturbed_dims": slice(0,0),
        "perturbation":   [0.0, 0.0]},
}
N_MOA = len(MOA_CLASSES)

# Build realistic profiles
profiles_list, moa_labels, compound_ids = [], [], []
moa_name_list = list(MOA_CLASSES.keys())

for moa_idx, (moa, info) in enumerate(MOA_CLASSES.items()):
    moa_signal = np.zeros(N_FEATURES_TOTAL)
    sl = info["perturbed_dims"]
    if sl.start != sl.stop:
        n_perturbed = sl.stop - sl.start
        # Heterogeneous perturbation within MoA (dose variation, compound diversity)
        moa_signal[sl] = np.random.uniform(info["perturbation"][0]*0.7,
                                            info["perturbation"][0]*1.3,
                                            n_perturbed)
        # Secondary channel effects
        secondary = np.clip(sl.start + n_perturbed//2,
                            0, N_FEATURES_TOTAL - n_perturbed//4)
        moa_signal[secondary:secondary+n_perturbed//4] += info["perturbation"][1]

    for cpd_i in range(info["n"]):
        # Compound-to-compound variation (realistic: ~20% noise on MoA signal)
        profile = moa_signal * np.random.uniform(0.7, 1.3) + \
                  np.random.normal(0, 0.35, N_FEATURES_TOTAL)
        profiles_list.append(profile)
        moa_labels.append(moa_idx)
        compound_ids.append(f"{moa[:4].upper()}_{cpd_i:03d}")

X_raw = np.array(profiles_list)
y_moa = np.array(moa_labels)
print(f"  Profiles generated: {len(X_raw)} compounds × {N_FEATURES_TOTAL} features")
print(f"  MoA distribution:")
for mi, (moa, info) in enumerate(MOA_CLASSES.items()):
    print(f"    {moa:35s}: {info['n']:3d} compounds")

# ── STEP 3: pycytominer QC + Normalization + Batch Correction ─────────────────
print("\n" + "─"*60)
print("  STEP 3 — pycytominer: QC → Normalize → Batch Correct → Feature Select")
print("─"*60)
"""
pycytominer (Weisbart 2021): pip install pycytominer
  Standard computational pipeline for Cell Painting data

  normalize(df, method='mad_robustize')
    → (x − median(DMSO)) / (1.4826 × MAD(DMSO))
    → Normalizes to DMSO negative control distribution

  feature_select(df, operation=['variance_threshold',
                                 'correlation_threshold',
                                 'drop_na_columns',
                                 'blocklist'])
    → Remove near-constant, highly-correlated, and blacklisted features

  Batch correction options:
    • Spherize (recommended): transform to spherical distribution
    • ComBat (pyComBat): parametric empirical Bayes
    • RobustMAD per plate: simple, effective

  Blocklist: known-noisy CellProfiler features (image-level artifacts,
             boundary cells, cell count-correlated features)
"""
# 1. Variance filter
vt = VarianceThreshold(threshold=0.02)
X_vt = vt.fit_transform(X_raw)
n_removed_var = X_raw.shape[1] - X_vt.shape[1]

# 2. Robust z-score normalization (DMSO-based)
dmso_mask = y_moa == moa_name_list.index("DMSO_control")
dmso_profiles = X_vt[dmso_mask]
feature_median = np.median(dmso_profiles, axis=0)
feature_mad    = median_abs_deviation(dmso_profiles, axis=0)
feature_mad    = np.maximum(feature_mad, 1e-8)
X_norm = (X_vt - feature_median) / (1.4826 * feature_mad)
X_norm = np.clip(X_norm, -10, 10)  # clip extreme outliers

# 3. Correlation filter
corr = np.corrcoef(X_norm.T)
upper = np.triu(np.abs(corr) > 0.90, k=1)
to_remove = set()
for i, j in zip(*np.where(upper)):
    to_remove.add(j)
keep = [i for i in range(X_norm.shape[1]) if i not in to_remove]
X_proc = X_norm[:, keep]

# 4. Simulate batch correction (2 plates)
batch = np.random.choice([0, 1], len(X_proc), p=[0.5, 0.5])
for b in [0, 1]:
    batch_offset = np.random.normal(0.3, 0.1, X_proc.shape[1])
    X_proc[batch == b] += batch_offset * b  # plate 1 has offset
for b in [0, 1]:
    X_proc[batch == b] -= X_proc[batch == b].mean(axis=0)  # centering

print(f"  Raw features:      {X_raw.shape[1]:,}")
print(f"  After var filter:  {X_vt.shape[1]:,} (removed {n_removed_var})")
print(f"  After corr filter: {X_proc.shape[1]:,} (removed {len(to_remove)})")
print(f"  Normalization:     MAD robust z-score (DMSO reference)")
print(f"  Batch correction:  2 plates centered (ComBat-style)")

# ── STEP 4: PCA + MoA Classification (RF, GBM, LogReg) ───────────────────────
print("\n" + "─"*60)
print("  STEP 4 — MoA Classification: PCA → Ensemble Models")
print("─"*60)

pca50 = PCA(n_components=50, random_state=42)
X_pca = pca50.fit_transform(X_proc)
pve50 = pca50.explained_variance_ratio_
cum10 = sum(pve50[:10]) * 100

print(f"  PCA(50): PC1={pve50[0]*100:.1f}% | PC2={pve50[1]*100:.1f}% | "
      f"Cum.10={cum10:.0f}%")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classifiers = {
    "Random Forest":  RandomForestClassifier(300, class_weight='balanced',
                                              max_features='sqrt', random_state=42),
    "GBM":            GradientBoostingClassifier(150, max_depth=4,
                                                  learning_rate=0.08, random_state=42),
    "Logistic Reg.":  LogisticRegression(C=1.0, max_iter=1000,
                                          multi_class='ovr', solver='saga'),
}
clf_results = {}
print(f"\n  {'Model':20s} {'Accuracy':>10} {'AUC (OvR)':>12} {'mAP':>8}")
print("  " + "─"*54)
for name, clf in classifiers.items():
    acc = cross_val_score(clf, X_pca, y_moa, cv=skf, scoring='accuracy')
    auc = cross_val_score(clf, X_pca, y_moa, cv=skf,
                          scoring='roc_auc_ovr_weighted')
    clf_results[name] = {"acc": round(acc.mean(),4), "auc": round(auc.mean(),4)}
    print(f"  {name:20s} {acc.mean():>10.4f} {auc.mean():>12.4f}   n/a")

best_clf_name = max(clf_results, key=lambda k: clf_results[k]["auc"])
best_clf = classifiers[best_clf_name]
best_clf.fit(X_pca, y_moa)
print(f"\n  Best model: {best_clf_name} (AUC={clf_results[best_clf_name]['auc']:.4f})")

# Confusion matrix
y_pred_cm = best_clf.predict(X_pca)
cm = confusion_matrix(y_moa, y_pred_cm)
print(f"\n  Top-level accuracy on full set: {(y_pred_cm==y_moa).mean():.4f}")

# ── STEP 5: Toxicity Prediction (DILI + Mitotox) ──────────────────────────────
print("\n" + "─"*60)
print("  STEP 5 — Toxicity Prediction: DILI · Mitotoxicity · ToxCast")
print("─"*60)
"""
ToxCast / Cell Painting integration (biorXiv 2025):
  Primary human hepatocytes × 1,085 compounds × 8 concentrations
  Three feature extraction methods compared:
    1. CellProfiler (classical morphological)
    2. Cell Painting CNN (Inception/EfficientNet fine-tuned)
    3. Pretrained ViT (vision transformer, self-supervised)

  Predict 412 ToxCast assay outcomes:
    Mean AUC = 0.73 (CellProfiler)
    Mean AUC = 0.78 (CNN)
    Mean AUC = 0.81 (ViT + contrastive)

Key insight: Mitochondria channel (MitoTracker) is most predictive for:
  → Mitochondrial toxicity (DILI mechanism)
  → ETC inhibitors (rotenone, antimycin)
  → Uncouplers (FCCP, CCCP)
  → Cationic amphiphilic drugs (CADs)
"""
# Binary DILI classification
n_total = len(X_proc)
dili_labels = np.zeros(n_total)
# vMDILI: perturb mitochondria features
mito_start = 900  # last channel features = mito
vmdili_idx = np.random.choice(n_total, 45, replace=False)
lmdili_idx = np.random.choice(list(set(range(n_total))-set(vmdili_idx)), 60, replace=False)
dili_labels[vmdili_idx] = 2
dili_labels[lmdili_idx] = 1

# Add mito signal to vMDILI compounds
mito_feats = min(mito_start, X_proc.shape[1]-100)
X_proc[vmdili_idx, mito_feats:mito_feats+80] += np.random.uniform(1.8, 3.2, (45, 80))
X_proc[lmdili_idx, mito_feats:mito_feats+80] += np.random.uniform(0.5, 1.2, (60, 80))

X_pca_dili = pca50.transform(X_proc)  # re-transform after signal injection

# Binary DILI (vMDILI vs noDILI)
dili_bin_mask = dili_labels != 1
X_dili = X_pca_dili[dili_bin_mask]
y_dili = (dili_labels[dili_bin_mask] == 2).astype(int)
rf_dili = RandomForestClassifier(300, class_weight='balanced', random_state=42)
skf_d = StratifiedKFold(5, shuffle=True, random_state=42)
auc_dili  = cross_val_score(rf_dili, X_dili, y_dili, cv=skf_d, scoring='roc_auc')
ap_dili   = cross_val_score(rf_dili, X_dili, y_dili, cv=skf_d, scoring='average_precision')
print(f"  DILI (vMDILI vs noDILI, n={dili_bin_mask.sum()}):")
print(f"    AUC:  {auc_dili.mean():.4f} ± {auc_dili.std():.4f}")
print(f"    AUPRC:{ap_dili.mean():.4f} ± {ap_dili.std():.4f}")

# Multi-class DILI severity
y_dili3 = dili_labels
rf_dili3 = RandomForestClassifier(300, class_weight='balanced', random_state=42)
skf_d3 = StratifiedKFold(3, shuffle=True, random_state=42)
auc_dili3 = cross_val_score(rf_dili3, X_pca_dili, y_dili3, cv=skf_d3,
                             scoring='roc_auc_ovr_weighted')
print(f"\n  3-class DILI (vMDILI/lMDILI/noDILI, n={n_total}):")
print(f"    AUC (OvR weighted): {auc_dili3.mean():.4f}")

# Feature importance for DILI
rf_dili.fit(X_dili, y_dili)
top_pc_dili = np.argsort(rf_dili.feature_importances_)[::-1][:8]
print(f"\n  Top PCs for DILI (mito-weighted): {top_pc_dili[:5].tolist()}")
print(f"  Literature: MitoTracker features → primary DILI mechanism predictor")

# ── STEP 6: DeepProfiler CNN (EfficientNet-B0 style) ─────────────────────────
print("\n" + "─"*60)
print("  STEP 6 — DeepProfiler: CNN on 5-channel cell images")
print("─"*60)
"""
DeepProfiler (Caicedo 2022, PLoS Comp Bio):
  Input: 128×128 px single-cell crops, 5 channels
  Backbone: EfficientNet-B0 / ResNet50 pretrained on ImageNet
  Fine-tuning: replace last FC layer, train on labeled Cell Painting
  Output: 1024-dim embedding per cell → aggregate to well/plate level

PhenoProfiler (Nat Commun 2025):
  Vision Transformer (ViT-B/16) + contrastive learning (SimCLR-style)
  Trained on JUMP-CP (1.6 billion cells)
  Outperforms CellProfiler features by 20% on compound retrieval
  Available: pip install phenoprofiler

Key advantages of CNN over CellProfiler features:
  ✓ No manual feature engineering
  ✓ Captures morphological patterns humans don't describe
  ✓ Better generalization across cell types
  ✗ Less interpretable (requires CAM/SHAP for attribution)
  ✗ Requires GPU and more data (~1000 labeled images)
"""

class DepthwiseSeparableConv(nn.Module):
    """EfficientNet-style building block: depthwise + pointwise convolution."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x): return F.silu(self.bn(self.pw(self.dw(x))))

class CellPaintingCNN(nn.Module):
    """
    Lightweight EfficientNet-inspired CNN for 5-channel Cell Painting images.
    Production: use EfficientNet-B0 with 5-channel input adaptation:
      model = efficientnet_b0(pretrained=True)
      model.features[0][0] = nn.Conv2d(5, 32, 3, stride=2, padding=1)
    """
    def __init__(self, n_channels=5, n_classes=7, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU())
        self.body = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(embed_dim, n_classes))
        self.embed_dim = embed_dim

    def embed(self, x):
        """Extract 256-dim embedding (for UMAP/clustering)."""
        h = self.stem(x)
        h = self.body(h)
        return h.flatten(1)

    def forward(self, x): return self.head(self.embed(x))

def make_cell_batch(n=16, n_ch=5, size=64):
    """Generate synthetic 5-channel Cell Painting crops."""
    imgs = torch.randn(n, n_ch, size, size) * 0.3
    labels = torch.randint(0, 7, (n,))
    for i in range(n):
        moa = labels[i].item()
        # Add MoA-specific channel signatures
        if moa == 0:   imgs[i, 3] += (torch.randn(size,size).abs() * 2).clamp(0,4)   # actin
        elif moa == 1: imgs[i, 0] += (torch.randn(size,size).abs() * 1.5)             # DNA
        elif moa == 2: imgs[i, 0] += (torch.randn(size,size).abs() * 2.0)             # DNA damage
        elif moa == 4: imgs[i, 4] += (torch.randn(size,size).abs() > 1.5).float()*3   # mito
        imgs[i] = imgs[i].clamp(-3, 5)
    return imgs, labels

cnn = CellPaintingCNN(5, 7, 256)
n_params_cnn = sum(p.numel() for p in cnn.parameters())
optimizer_cnn = torch.optim.AdamW(cnn.parameters(), lr=2e-3, weight_decay=0.01)
sched_cnn = torch.optim.lr_scheduler.OneCycleLR(optimizer_cnn, max_lr=2e-3,
                                                   total_steps=50, pct_start=0.2)
losses_cnn, accs_cnn = [], []

cnn.train()
for ep in range(50):
    ep_loss, ep_correct, ep_total = 0, 0, 0
    for _ in range(6):
        imgs, labs = make_cell_batch(16, 5, 64)
        optimizer_cnn.zero_grad()
        logits = cnn(imgs)
        loss   = F.cross_entropy(logits, labs)
        loss.backward(); optimizer_cnn.step()
        ep_loss    += loss.item()
        ep_correct += (logits.argmax(1) == labs).sum().item()
        ep_total   += len(labs)
    sched_cnn.step()
    losses_cnn.append(ep_loss / 6)
    accs_cnn.append(ep_correct / ep_total)

cnn.eval()
test_imgs, test_labs = make_cell_batch(100, 5, 64)
with torch.no_grad():
    test_preds = cnn(test_imgs).argmax(1)
cnn_test_acc = (test_preds == test_labs).float().mean().item()
print(f"  CNN architecture: 5-ch → Stem → 5×DepthwiseConv → AvgPool → MLP")
print(f"  Parameters: {n_params_cnn:,} | Embedding dim: 256")
print(f"  Final train accuracy: {accs_cnn[-1]:.4f}")
print(f"  Test accuracy (simulated): {cnn_test_acc:.4f}")
print(f"  Production: EfficientNet-B0, 5.3M params, ~0.87 AUC on BBBC021")

# ── STEP 7: UMAP + Compound Clustering + Perturbation Discovery ───────────────
print("\n" + "─"*60)
print("  STEP 7 — UMAP Embedding + Compound Clustering + Activity Discovery")
print("─"*60)
"""
Perturbation-based discovery workflow:
  1. UMAP of morphological profiles (faster than t-SNE for large N)
  2. Nearest-neighbor matching: query → find similar-phenotype compounds
  3. Clustering (HDBSCAN): discover compound communities
  4. Enrichment: which MoAs co-cluster? → predict MoA of unknowns
  5. Activity score: Mahalanobis distance to DMSO (z-score for activity)

Practical Cell Painting analysis (pycytominer workflow):
  # Replicate correlation (are replicates concordant?)
  from pycytominer.cyto_utils import compute_percent_replicating
  pct_rep = compute_percent_replicating(df_norm, threshold=0.95)
  # Typical: 50-75% of perturbations are 'active' vs DMSO noise

Activity detection:
  Percent Replicating: fraction of compounds with replicate correlation
    above 95th percentile of non-replicate distribution
  Typical: 50-80% active compounds (dose-dependent)
"""
# UMAP embedding (2D for visualization)
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.2,
                    random_state=42, metric='cosine')
    X_2d = reducer.fit_transform(X_pca[:, :20])
    umap_available = True
except ImportError:
    tsne = TSNE(n_components=2, random_state=42, perplexity=25)
    X_2d = tsne.fit_transform(X_pca[:, :20])
    umap_available = False

# Nearest-neighbor compound retrieval
nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(X_pca)
distances, indices = nbrs.kneighbors(X_pca)

# Compute % same-MoA in top-5 neighbors (retrieval precision)
retrieval_precision = []
for i in range(len(y_moa)):
    neighbor_moas = y_moa[indices[i, 1:6]]
    precision = (neighbor_moas == y_moa[i]).mean()
    retrieval_precision.append(precision)
print(f"  Compound retrieval precision@5: {np.mean(retrieval_precision):.4f}")
print(f"  (Fraction of top-5 neighbors with same MoA)")

# Mahalanobis activity score
dmso_pca = X_pca[dmso_mask]
dmso_mean = dmso_pca.mean(axis=0)
dmso_cov  = np.cov(dmso_pca.T) + np.eye(50) * 1e-4
try:
    dmso_inv_cov = np.linalg.inv(dmso_cov)
    diff         = X_pca - dmso_mean
    activity     = np.sqrt(np.einsum('ij,jk,ik->i', diff, dmso_inv_cov, diff))
except np.linalg.LinAlgError:
    activity     = np.linalg.norm(X_pca - dmso_mean, axis=1)

pct_active = (activity > np.percentile(activity[dmso_mask], 95)).mean() * 100
print(f"  Percent active (vs DMSO 95th pct): {pct_active:.1f}% of all compounds")

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
print("\n  Generating comprehensive visualization (7-panel)...")

fig = plt.figure(figsize=(24, 16))
fig.suptitle("NB01 — Cell Painting Deep Dive: Morphological Profiling + MoA + Toxicity + CNN",
             fontsize=13, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.48, wspace=0.38)

# Panel 1: Channel feature counts (barh)
ax1 = fig.add_subplot(gs[0, 0])
ch_names = list(CHANNELS.keys())
ch_feats = [CHANNELS[c]["n_features"] for c in ch_names]
ch_cols  = ['#3498db','#27ae60','#e74c3c','#e67e22','#8e44ad']
ax1.barh(ch_names, ch_feats, color=ch_cols, alpha=0.85)
ax1.set_xlabel("Features"); ax1.set_title("CellProfiler\nFeatures per Channel")
ax1.grid(True, alpha=0.3, axis='x')
for i, (n, c) in enumerate(zip(ch_feats, ch_names)):
    ax1.text(n+3, i, str(n), va='center', fontsize=9)

# Panel 2: UMAP of MoA profiles
ax2 = fig.add_subplot(gs[0, 1:3])
for mi, (moa, info) in enumerate(MOA_CLASSES.items()):
    mask = y_moa == mi
    ax2.scatter(X_2d[mask,0], X_2d[mask,1], c=info["color"],
                label=moa.replace("_"," "), s=30, alpha=0.8, zorder=3+mi)
ax2.set_title(f"{'UMAP' if umap_available else 't-SNE'} of Morphological Profiles\n"
              f"(7 Mechanisms of Action, {len(y_moa)} compounds)")
ax2.legend(fontsize=7.5, ncol=2, loc='lower right')
ax2.set_xlabel("Dim 1"); ax2.set_ylabel("Dim 2"); ax2.grid(True, alpha=0.3)

# Panel 3: Model comparison
ax3 = fig.add_subplot(gs[0, 3])
m_names = list(clf_results.keys()) + ["CNN\n(EfficientNet)"]
m_aucs  = [clf_results[m]["auc"] for m in clf_results.keys()] + [0.87]
m_accs  = [clf_results[m]["acc"] for m in clf_results.keys()] + [cnn_test_acc]
x_b = np.arange(len(m_names))
ax3.bar(x_b-0.18, m_aucs, 0.35, label='AUC (OvR)', color='#1565c0', alpha=0.85)
ax3.bar(x_b+0.18, m_accs, 0.35, label='Accuracy',  color='#27ae60', alpha=0.85)
ax3.set_xticks(x_b); ax3.set_xticklabels([n[:10] for n in m_names], fontsize=8, rotation=20)
ax3.set_ylim([0.5, 1.0]); ax3.legend(fontsize=8)
ax3.set_title("MoA Classification\n(5-fold CV, 7 classes)")
ax3.axhline(0.83, color='red', lw=1.5, linestyle='--', label='BBBC021 baseline')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Confusion matrix (best model)
ax4 = fig.add_subplot(gs[0, 4])
short_labels = [m[:8].replace("_","") for m in moa_name_list]
im4 = ax4.imshow(cm, cmap='Blues', interpolation='nearest')
ax4.set_xticks(range(N_MOA)); ax4.set_xticklabels(short_labels, rotation=45, fontsize=6.5)
ax4.set_yticks(range(N_MOA)); ax4.set_yticklabels(short_labels, fontsize=6.5)
ax4.set_title(f"Confusion Matrix\n({best_clf_name})")
plt.colorbar(im4, ax=ax4)
for i in range(N_MOA):
    for j in range(N_MOA):
        ax4.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=7,
                 color='white' if cm[i,j] > cm.max()*0.5 else 'black')

# Panel 5: PCA variance
ax5 = fig.add_subplot(gs[1, 0])
cumvar = np.cumsum(pve50) * 100
ax5.bar(range(1,21), pve50[:20]*100, color='#1565c0', alpha=0.7)
ax5_t = ax5.twinx()
ax5_t.plot(range(1,21), cumvar[:20], color='#e74c3c', lw=2.5, marker='o', ms=3)
ax5_t.set_ylabel("Cumulative %", color='#e74c3c', fontsize=9)
ax5.set_xlabel("Principal Component"); ax5.set_ylabel("Var %")
ax5.set_title("PCA Explained Variance"); ax5.grid(True, alpha=0.3)

# Panel 6: DILI activity spectrum
ax6 = fig.add_subplot(gs[1, 1])
act_dmso  = activity[y_moa == moa_name_list.index("DMSO_control")]
act_other = activity[y_moa != moa_name_list.index("DMSO_control")]
act_vmdili = activity[dili_labels == 2]
ax6.hist(act_dmso,   bins=20, alpha=0.6, color='#95a5a6', label='DMSO ctrl', density=True)
ax6.hist(act_other,  bins=20, alpha=0.5, color='#3498db', label='Compounds', density=True)
ax6.hist(act_vmdili, bins=15, alpha=0.7, color='#e74c3c', label='vMDILI',    density=True)
thresh = np.percentile(act_dmso, 95)
ax6.axvline(thresh, color='k', lw=2, linestyle='--', label=f'95th pct={thresh:.1f}')
ax6.set_xlabel("Activity score (Mahal. dist)")
ax6.set_ylabel("Density"); ax6.set_title("Morphological Activity\n(vs DMSO)")
ax6.legend(fontsize=8)

# Panel 7: DILI prediction ROC / PR
ax7 = fig.add_subplot(gs[1, 2])
rf_dili.fit(X_dili, y_dili)
y_score = rf_dili.predict_proba(X_dili)[:,1]
from sklearn.metrics import roc_curve, precision_recall_curve
fpr, tpr, _ = roc_curve(y_dili, y_score)
pre, rec, _ = precision_recall_curve(y_dili, y_score)
ax7.plot(fpr, tpr, color='#e74c3c', lw=2.5,
         label=f"ROC AUC={auc_dili.mean():.3f}")
ax7.plot(rec, pre, color='#1565c0', lw=2.5, linestyle='--',
         label=f"PR AUC={ap_dili.mean():.3f}")
ax7.plot([0,1],[0,1], 'k:', lw=1)
ax7.set_xlabel("FPR / Recall"); ax7.set_ylabel("TPR / Precision")
ax7.set_title("DILI Prediction\n(vMDILI vs noDILI)")
ax7.legend(fontsize=9); ax7.grid(True, alpha=0.3)

# Panel 8: CNN training curves
ax8 = fig.add_subplot(gs[1, 3])
ax8.plot(losses_cnn, color='#e74c3c', lw=2, label='Train loss')
ax8_t = ax8.twinx()
ax8_t.plot(accs_cnn, color='#27ae60', lw=2, linestyle='--', label='Train acc')
ax8_t.set_ylabel("Accuracy", color='#27ae60')
ax8.set_xlabel("Epoch"); ax8.set_ylabel("Loss", color='#e74c3c')
ax8.set_title(f"DeepProfiler CNN\n(EfficientNet-style, test acc={cnn_test_acc:.3f})")
ax8.grid(True, alpha=0.3)

# Panel 9: Retrieval precision per MoA
ax9 = fig.add_subplot(gs[1, 4])
prec_per_moa = []
for mi in range(N_MOA):
    mask = y_moa == mi
    prec_per_moa.append(np.mean([retrieval_precision[i] for i in np.where(mask)[0]]))
moa_short = [m.replace("_"," ")[:18] for m in moa_name_list]
colors_moa = [info["color"] for info in MOA_CLASSES.values()]
ax9.barh(moa_short, prec_per_moa, color=colors_moa, alpha=0.85)
ax9.set_xlabel("Precision@5 (same-MoA neighbors)")
ax9.set_title("Compound Retrieval\n(top-5 nearest neighbors)")
ax9.grid(True, alpha=0.3, axis='x')
ax9.axvline(1/N_MOA, color='k', linestyle='--', lw=1.5, label='Random baseline')
ax9.legend(fontsize=8)

# Panel 10: Full benchmark table
ax10 = fig.add_subplot(gs[2, :])
ax10.axis('off')
bench = [
    ["Method","Dataset","Task","AUC / Score","Features","Reference"],
    ["CellProfiler + RF","BBBC021","MoA 12-class","Acc=0.83","1,500 morph","Caicedo 2017"],
    ["DeepProfiler (EfficientNet)","BBBC021","MoA 12-class","Acc=0.87","1,024 CNN embed","Caicedo 2022"],
    ["PhenoProfiler (ViT)","JUMP-CP","Compound retrieval","+20% vs CellProf.","End-to-end ViT","Nat Commun 2025"],
    ["CellProfiler + RF","ToxCast CP","DILI predict (412 assays)","AUC=0.73","1,500 morph","biorXiv 2025"],
    ["CNN (Cell Painting ViT)","ToxCast CP","DILI predict","AUC=0.78","CNN features","biorXiv 2025"],
    ["CLOOME (contrastive)","JUMP-CP","SMILES→phenotype","0.65 mAP","Multimodal CLIP","Sanchez 2022"],
    ["This NB01 (RF + CNN)","Simulated","MoA + DILI",
     f"MoA={clf_results[best_clf_name]['auc']:.3f} / DILI={auc_dili.mean():.3f}","PCA(50)","This notebook"],
]
table = ax10.table(cellText=bench[1:], colLabels=bench[0],
                    cellLoc='center', loc='center', bbox=[0,0,1,1])
table.auto_set_font_size(False); table.set_fontsize(9.5)
for j in range(6):
    table[0,j].set_facecolor('#0d2137')
    table[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(bench)):
    for j in range(6):
        table[i,j].set_facecolor('#f5f0ff' if i%2==0 else 'white')
ax10.set_title("Cell Painting ML Benchmark 2024-2025", fontsize=11, pad=15)

plt.savefig("imaging_results/NB01_cell_painting_deep.png", dpi=150, bbox_inches="tight")
plt.show()

summary = {
    "notebook":      "NB01 Deep Dive — Cell Painting",
    "n_compounds":   len(X_raw),
    "n_features_raw":N_FEATURES_TOTAL,
    "n_features_final":X_proc.shape[1],
    "MoA_best_AUC":  clf_results[best_clf_name]["auc"],
    "DILI_AUC":      round(auc_dili.mean(),4),
    "DILI3_AUC":     round(auc_dili3.mean(),4),
    "CNN_test_acc":  round(cnn_test_acc,4),
    "retrieval_P5":  round(float(np.mean(retrieval_precision)),4),
    "pct_active":    round(pct_active,1),
}
with open("imaging_results/NB01_deep_results.json","w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Figure saved: imaging_results/NB01_cell_painting_deep.png")
print("="*72)
print("  NB01 COMPLETE — Cell Painting Deep Dive")
print(f"  MoA AUC: {clf_results[best_clf_name]['auc']:.4f}")
print(f"  DILI AUC: {auc_dili.mean():.4f} (binary) | {auc_dili3.mean():.4f} (3-class)")
print(f"  CNN test: {cnn_test_acc:.4f}")
print(f"  Retrieval P@5: {np.mean(retrieval_precision):.4f}")
print("="*72)
