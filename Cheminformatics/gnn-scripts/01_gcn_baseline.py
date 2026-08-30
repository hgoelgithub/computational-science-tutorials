"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GNN Script 01 — Basic Graph Convolutional Network (GCN) Baseline           ║
║  Task: Molecular property prediction (hERG cardiotoxicity, IC50 binary)     ║
║  Author: Himanshu Goel | hgoelgithub.github.io                              ║
║                                                                              ║
║  Architecture: Kipf & Welling (2017) GCN                                    ║
║  Message passing: h_v = ReLU(W · MEAN(h_u for u in N(v) ∪ {v}))            ║
║  Pooling: Global mean pooling → MLP classifier                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

LEARNING OBJECTIVES
───────────────────
1. Understand how molecules are represented as graphs
2. Build atom (node) and bond (edge) feature matrices from scratch
3. Implement vanilla GCN message passing using PyTorch Geometric
4. Evaluate with ROC-AUC using scaffold-aware splitting
5. Understand limitations that motivate more advanced GNNs (Scripts 2-5)

BACKGROUND
──────────
Molecules are natural graphs:
  - Nodes = atoms  (features: atomic number, degree, charge, aromaticity ...)
  - Edges = bonds  (features: bond type, conjugation, ring membership ...)

GCN aggregates neighbor features via normalized adjacency:
  H' = σ(D^{-1/2} A D^{-1/2} H W)

where A = adjacency + self-loops, D = degree matrix, W = learned weight.

Key limitation: GCN uses symmetric normalization → cannot distinguish
node degrees → expressiveness bounded by 1-WL graph isomorphism test.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    HAS_PYG = True
except ImportError:
    print("PyTorch Geometric not installed. Install with:")
    print("  pip install torch-geometric")
    HAS_PYG = False

# Scikit-learn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "task":         "hERG_binary",        # molecular property to predict
    "n_epochs":     80,
    "batch_size":   32,
    "lr":           1e-3,
    "hidden_dim":   64,
    "n_layers":     3,
    "dropout":      0.3,
    "seed":         42,
    "split":        "scaffold",           # scaffold | random
    "train_frac":   0.8,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("="*70)
print("  GNN Script 01 — Basic GCN Baseline")
print("="*70)
print(f"  Device : {CONFIG['device']}")
print(f"  Task   : {CONFIG['task']}")
print(f"  Split  : {CONFIG['split']}")

# ── Dataset: hERG cardiotoxicity ──────────────────────────────────────────────
"""
hERG (human Ether-a-go-go Related Gene) encodes a cardiac K+ channel.
Blocking it causes QT prolongation → potentially fatal arrhythmia.
Multiple drugs were withdrawn due to unexpected hERG blockade.

Binary label: 1 = hERG blocker (IC50 < 10 uM), 0 = non-blocker
Source: curated from ChEMBL (hERG assay: CHEMBL240)
"""
HERG_DATA = [
    # (SMILES, label, compound_name)
    ("OC(c1ccc(C(c2ccccc2)(c2ccccc2)O)cc1)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1", 1, "Terfenadine"),
    ("CCOC(=O)c1cc2cc(OC)c(OC)cc2[nH]1", 1, "Cisapride"),
    ("CN(CCOc1ccc(NS(=O)(=O)c2ccc(NC)cc2)cc1)S(=O)(=O)c1ccc(N)cc1", 1, "Dofetilide"),
    ("Clc1ccc2c(c1)n(CCN1CCC(=C3c4cc(F)ccc4NC3=O)CC1)c(=O)n2", 1, "Sertindole"),
    ("Fc1ccc(CC2CCN(CCc3ccc(F)cc3F)CC2)cc1", 1, "Haloperidol-analog"),
    ("COc1ccc(CCN(C)CCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", 1, "Verapamil"),
    ("OC(c1ccnc2ccccc12)C1CC2CCN1CC2C=C", 1, "Quinidine"),
    ("CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21", 1, "Chlorpromazine"),
    ("OCC(NC(=O)c1nc2cc(OCC(F)(F)F)ccc2c(OCC(F)(F)F)c1)C", 1, "Flecainide"),
    ("Cn1cc2c(cn1)CC(=O)N2CC1CCNCC1", 1, "Ondansetron-analog"),
    ("CCCc1nc2ccccc2c(=O)n1C", 1, "Phenobarbital-analog"),
    ("c1ccc2c(c1)n(CCN1CCCCC1)c(=O)n2", 1, "Imipramine-analog"),
    ("CC(O)CNc1ccc(NS(C)(=O)=O)cc1", 0, "Sotalol"),
    ("COc1ccc(OCC(O)CN2CC(=O)N(c3ccccc3F)CC2)cc1OC", 0, "Ranolazine"),
    ("CC(=O)Oc1ccccc1C(=O)O", 0, "Aspirin"),
    ("CN(C)C(=N)NC(=N)N", 0, "Metformin"),
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 0, "Caffeine"),
    ("OCC(O)CO", 0, "Glycerol"),
    ("OC(=O)c1ccccc1", 0, "Benzoic acid"),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0, "Ibuprofen"),
    ("CC(O)CNc1ccc(O)cc1", 0, "Salbutamol-analog"),
    ("NC(=O)c1ccc[n+](C2OC(CO)C(O)C2O)c1", 0, "Nicotinamide riboside"),
    ("Oc1ccc(CC(N)Cc2ccc(O)cc2)cc1", 0, "Tyrosine analog"),
    ("CC(=O)Nc1ccc(O)cc1", 0, "Acetaminophen"),
    ("OC(=O)CC(O)(CC(=O)O)C(=O)O", 0, "Citric acid"),
    ("CC(C)(C)c1ccc(O)cc1", 0, "4-tBu-phenol"),
    ("OC1=CC=C2CC3N(CCC34CCc5c4cc(O)c(OC)c5)C2=C1", 1, "Morphine"),
    ("CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1", 0, "Fluoxetine"),
    ("CC(=O)OCC", 0, "Ethyl acetate"),
    ("CC(C)NCC(O)COc1cccc2ccccc12", 0, "Propranolol"),
    ("CC(=O)Nc1ccc(NS(=O)(=O)c2ccc(N)cc2)cc1", 0, "Dapsone"),
    ("Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1", 0, "Sulfadiazine"),
    ("OC(=O)c1ccc(Cl)cc1", 0, "4-Chlorobenzoic acid"),
    ("c1ccc(Cl)c(Cl)c1", 0, "1,2-Dichlorobenzene"),
    ("OCC(O)C(O)C(O)CO", 0, "Xylitol"),
    ("c1ccc2ncccc2c1", 0, "Quinoline"),
    ("NC(CS)C(=O)O", 0, "Cysteine"),
    ("CC(N)Cc1ccccc1", 0, "Amphetamine"),
    ("OC(=O)CCc1ccccc1", 0, "Hydrocinnamic acid"),
    ("Nc1ccc([N+](=O)[O-])cc1", 0, "4-Nitroaniline"),
]

# ── Atom & Bond Featurization ─────────────────────────────────────────────────
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree":     [0,1,2,3,4,5,6,7,8,9,10],
    "formal_charge": [-5,-4,-3,-2,-1,0,1,2,3,4,5],
    "num_hs":     [0,1,2,3,4,5,6,7,8],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

def one_hot(val, choices, allow_unknown=True):
    """One-hot encode val against choices list."""
    enc = [0] * (len(choices) + (1 if allow_unknown else 0))
    if val in choices:
        enc[choices.index(val)] = 1
    elif allow_unknown:
        enc[-1] = 1   # 'other' bucket
    return enc

def atom_features(atom):
    """
    Build atom feature vector (39-dimensional):
      - atomic number (118-dim one-hot)  → compressed to 11 most common + 'other'
      - degree (0-10)
      - formal charge (-5 to +5)
      - num implicit Hs (0-8)
      - hybridization (sp, sp2, sp3, sp3d, sp3d2, 'other')
      - aromaticity (1-bit)
      - is in ring (1-bit)
    """
    common_atoms = [1,6,7,8,9,15,16,17,35,53,14]   # H,C,N,O,F,P,S,Cl,Br,I,Si
    feats = (
        one_hot(atom.GetAtomicNum(), common_atoms)           +  # 12-dim
        one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])   +  # 12-dim
        one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"]) + # 12-dim
        one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"]) +  # 10-dim
        one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"]) + # 6-dim
        [int(atom.GetIsAromatic())]                          +  # 1-dim
        [int(atom.IsInRing())]                                  # 1-dim
    )
    return feats  # total: 54-dim

def bond_features(bond):
    """
    Build bond feature vector (10-dimensional):
      - bond type (single, double, triple, aromatic)
      - conjugated (1-bit)
      - in ring (1-bit)
      - stereo (none, any, Z, E) → 4-dim
    """
    bt = bond.GetBondTypeAsDouble()
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    feats = (
        one_hot(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
        ])                                                    +  # 5-dim
        [int(bond.GetIsConjugated())]                         +  # 1-dim
        [int(bond.IsInRing())]                                +  # 1-dim
        one_hot(bond.GetStereo(), stereo_types)                  # 5-dim
    )
    return feats  # total: 12-dim

def mol_to_graph(smiles, label):
    """Convert SMILES string → PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge indices and features (bidirectional)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]   # same features for both directions

    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 12), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# ── Scaffold Splitter ─────────────────────────────────────────────────────────
def scaffold_split(smiles_list, labels, train_frac=0.8, seed=42):
    """
    Scaffold-based train/test split (Bemis-Murcko scaffolds).
    Molecules sharing the same scaffold go to the same split.
    This is more realistic than random split for drug discovery.
    """
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        else:
            scaffold = smi   # fallback
        scaffolds[scaffold].append(idx)

    rng = np.random.RandomState(seed)
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)

    train_idx, test_idx = [], []
    n_train = int(len(smiles_list) * train_frac)
    for s_set in scaffold_sets:
        if len(train_idx) < n_train:
            train_idx.extend(s_set)
        else:
            test_idx.extend(s_set)

    return train_idx, test_idx

# ── Build Dataset ─────────────────────────────────────────────────────────────
print("\n[1/5] Building molecular graph dataset...")
dataset = []
failed  = []
for smi, label, name in HERG_DATA:
    g = mol_to_graph(smi, label)
    if g is not None:
        g.smiles = smi
        g.name   = name
        dataset.append(g)
    else:
        failed.append(name)

print(f"  Molecules parsed : {len(dataset)}/{len(HERG_DATA)}")
print(f"  Failed           : {failed if failed else 'None'}")
print(f"  Node feature dim : {dataset[0].x.shape[1]}")
print(f"  Edge feature dim : {dataset[0].edge_attr.shape[1] if dataset[0].edge_attr.numel() > 0 else 'N/A'}")
print(f"  Positive (blocker): {sum(1 for g in dataset if g.y.item()==1)}")
print(f"  Negative (safe)   : {sum(1 for g in dataset if g.y.item()==0)}")

N_NODE_FEAT = dataset[0].x.shape[1]

# Split
smiles_all = [g.smiles for g in dataset]
labels_all  = [int(g.y.item()) for g in dataset]

if CONFIG["split"] == "scaffold":
    train_idx, test_idx = scaffold_split(smiles_all, labels_all,
                                          CONFIG["train_frac"], CONFIG["seed"])
else:
    rng = np.random.RandomState(CONFIG["seed"])
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    n_train = int(len(idx) * CONFIG["train_frac"])
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

train_data = [dataset[i] for i in train_idx]
test_data  = [dataset[i] for i in test_idx]

print(f"\n  Split ({CONFIG['split']}): {len(train_data)} train / {len(test_data)} test")

train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=CONFIG["batch_size"], shuffle=False)

# ── GCN Model ─────────────────────────────────────────────────────────────────
class GCNMolecular(nn.Module):
    """
    Vanilla Graph Convolutional Network for molecular property prediction.

    Architecture:
      Input (node features)
        → GCNConv(hidden) → BN → ReLU → Dropout
        → GCNConv(hidden) → BN → ReLU → Dropout
        → GCNConv(hidden) → BN → ReLU
        → Global Mean Pool  (graph-level embedding)
        → MLP(hidden, hidden//2, 1)
        → Sigmoid

    Note: GCNConv does NOT use edge features — this is a key limitation.
    Edge features are added in Script 02 (MPNN) and Script 03 (GAT).
    """
    def __init__(self, in_channels, hidden_channels, n_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # Message passing layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Graph-level pooling: aggregate all node embeddings
        x = global_mean_pool(x, batch)   # shape: [n_graphs, hidden_channels]

        # Classification head
        out = self.mlp(x)
        return out.squeeze(-1)

    def get_graph_embeddings(self, x, edge_index, batch):
        """Extract graph-level embeddings (for visualization)."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch)
        loss  = F.binary_cross_entropy_with_logits(out, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        probs  = torch.sigmoid(out).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
        labels = batch.y.squeeze().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist() if labels.ndim > 0 else [labels.tolist()])
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    ap  = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, auc, ap, all_probs, all_labels

# ── Run Training ──────────────────────────────────────────────────────────────
if not HAS_PYG:
    print("\n[ERROR] PyTorch Geometric required. Skipping training.")
    exit()

device = torch.device(CONFIG["device"])
model  = GCNMolecular(
    in_channels      = N_NODE_FEAT,
    hidden_channels  = CONFIG["hidden_dim"],
    n_layers         = CONFIG["n_layers"],
    dropout          = CONFIG["dropout"],
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[2/5] Model created")
print(f"  Architecture    : {CONFIG['n_layers']}-layer GCN")
print(f"  Hidden dim      : {CONFIG['hidden_dim']}")
print(f"  Trainable params: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)

print(f"\n[3/5] Training for {CONFIG['n_epochs']} epochs...")
history = {"train_loss": [], "test_loss": [], "test_auc": [], "test_ap": [], "lr": []}
best_auc   = 0.0
best_state = None
t_start    = time.time()

for epoch in range(1, CONFIG["n_epochs"] + 1):
    tr_loss = train_epoch(model, train_loader, optimizer, device)
    te_loss, te_auc, te_ap, _, _ = evaluate(model, test_loader, device)
    scheduler.step(te_auc)
    lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(tr_loss)
    history["test_loss"].append(te_loss)
    history["test_auc"].append(te_auc)
    history["test_ap"].append(te_ap)
    history["lr"].append(lr)

    if te_auc > best_auc:
        best_auc   = te_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | Train loss: {tr_loss:.4f} | "
              f"Test AUC: {te_auc:.4f} | AP: {te_ap:.4f} | LR: {lr:.2e}")

t_train = time.time() - t_start
print(f"\n  Training time   : {t_train:.1f}s")
print(f"  Best test AUC   : {best_auc:.4f}")

# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n[4/5] Final evaluation on test set...")
model.load_state_dict(best_state)
_, final_auc, final_ap, final_probs, final_labels = evaluate(model, test_loader, device)
final_preds = (np.array(final_probs) > 0.5).astype(int)

print(f"\n  ROC-AUC                : {final_auc:.4f}")
print(f"  Average Precision (AP) : {final_ap:.4f}")
print(f"\n  Classification Report:")
print(classification_report(final_labels, final_preds,
                             target_names=["Non-blocker", "hERG blocker"],
                             zero_division=0))

# Save results for comparison in Script 5
results_summary = {
    "model":     "GCN_baseline",
    "n_params":  n_params,
    "best_auc":  round(best_auc, 4),
    "final_auc": round(final_auc, 4),
    "final_ap":  round(final_ap, 4),
    "train_time_s": round(t_train, 1),
    "config":    CONFIG,
}
os.makedirs("gnn_results", exist_ok=True)
with open("gnn_results/01_gcn_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# ── Visualization ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating visualizations...")
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Script 01 — Basic GCN: hERG Cardiotoxicity Prediction",
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1: Training curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history["train_loss"], label="Train loss", color="#1565c0", lw=2)
ax1.plot(history["test_loss"],  label="Test loss",  color="#e65100",  lw=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
ax1.set_title("Training Curves"); ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: AUC over epochs
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history["test_auc"], color="#27ae60", lw=2, label="Test ROC-AUC")
ax2.plot(history["test_ap"],  color="#8e44ad", lw=2, linestyle="--", label="Test AP")
ax2.axhline(best_auc, color="red", linestyle=":", lw=1.5, label=f"Best AUC={best_auc:.3f}")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric")
ax2.set_title("AUC / AP Over Training"); ax2.legend(fontsize=8)
ax2.set_ylim([0, 1]); ax2.grid(True, alpha=0.3)

# Panel 3: ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(final_labels, final_probs)
ax3.plot(fpr, tpr, color="#1565c0", lw=2.5, label=f"GCN (AUC={final_auc:.3f})")
ax3.plot([0,1],[0,1], "k--", lw=1, label="Random")
ax3.fill_between(fpr, tpr, alpha=0.1, color="#1565c0")
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve — hERG Prediction"); ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Confusion matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(final_labels, final_preds)
im = ax4.imshow(cm, cmap="Blues")
ax4.set_xticks([0,1]); ax4.set_xticklabels(["Safe","Blocker"])
ax4.set_yticks([0,1]); ax4.set_yticklabels(["Safe","Blocker"])
ax4.set_xlabel("Predicted"); ax4.set_ylabel("True")
ax4.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax4.text(j, i, cm[i,j], ha='center', va='center',
                 fontsize=14, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax4, fraction=0.046)

# Panel 5: Predicted probability distribution
ax5 = fig.add_subplot(gs[1, 1])
probs_arr = np.array(final_probs)
labels_arr = np.array(final_labels)
ax5.hist(probs_arr[labels_arr==0], bins=15, alpha=0.6, color="#27ae60",
         label="True safe", density=True)
ax5.hist(probs_arr[labels_arr==1], bins=15, alpha=0.6, color="#e74c3c",
         label="True blocker", density=True)
ax5.axvline(0.5, color='k', linestyle='--', lw=1.5, label="Threshold 0.5")
ax5.set_xlabel("Predicted probability of hERG blockade")
ax5.set_ylabel("Density"); ax5.set_title("Score Distribution")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# Panel 6: Model architecture summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary_text = (
    "GCN Architecture\n"
    "─────────────────────────────\n"
    f"Node features  : {N_NODE_FEAT} dim\n"
    "Edge features  : NOT USED ✗\n"
    f"Hidden dim     : {CONFIG['hidden_dim']}\n"
    f"GCN layers     : {CONFIG['n_layers']}\n"
    f"Trainable params: {n_params:,}\n"
    "Pooling        : Global Mean\n"
    "─────────────────────────────\n"
    f"Best AUC       : {best_auc:.4f}\n"
    f"Final AUC      : {final_auc:.4f}\n"
    f"Train time     : {t_train:.1f}s\n"
    "─────────────────────────────\n"
    "Limitation: Cannot use\n"
    "edge features or distinguish\n"
    "node degrees (WL-bounded)\n"
    "→ See Script 02 (MPNN)\n"
    "   Script 03 (GAT)\n"
    "   Script 04 (GIN+Attention)\n"
    "   Script 05 (Comparison)"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=8.5, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))

plt.savefig("gnn_results/01_gcn_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved: gnn_results/01_gcn_results.png")
print("  JSON saved: gnn_results/01_gcn_results.json")
print("\n" + "="*70)
print("  Script 01 complete. Key points:")
print("  - GCN uses symmetric normalized adjacency (Kipf & Welling 2017)")
print("  - Edge features (bond type, stereo) are IGNORED in GCN")
print("  - Expressiveness bounded by 1-WL isomorphism test")
print("  - Next: Script 02 adds edge features via MPNN framework")
print("="*70)
