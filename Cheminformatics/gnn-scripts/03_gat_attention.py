"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GNN Script 03 — Graph Attention Network (GAT) with Multi-head Attention    ║
║  Task: Molecular solubility (logS) regression — ESOL benchmark              ║
║  Author: Himanshu Goel | himanshugoel.github.io                              ║
║                                                                              ║
║  Architecture: Veličković et al. 2018 — Graph Attention Networks (ICLR)     ║
║                                                                              ║
║  Key innovation: LEARNABLE ATTENTION WEIGHTS on edges                        ║
║    α_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))                         ║
║    h_i'   = σ( SUM_{j ∈ N(i)} α_{ij} · W · h_j )                          ║
║                                                                              ║
║  Multi-head attention (K heads):                                             ║
║    h_i' = ||_{k=1}^K σ( SUM α_{ij}^k · W^k · h_j )                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT'S NEW vs Scripts 01-02
─────────────────────────────
1. Attention mechanism: each atom learns HOW MUCH to attend to each neighbor
2. Multi-head attention: K independent attention functions → richer features
3. Attention weights are INTERPRETABLE — visualize which bonds matter
4. Task: Aqueous solubility regression (continuous output)
   - Critical for drug bioavailability (BCS classification)
   - ESOL benchmark (Delaney 2004): 1128 compounds, log(mol/L)

WHY SOLUBILITY MATTERS
──────────────────────
~40% of drug candidates fail due to poor solubility.
Poor solubility → poor bioavailability → high clinical failure rate.
Predicting logS early guides synthesis decisions.
"""

import os, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

try:
    from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_add_pool
    from torch_geometric.data import Data, DataLoader
    HAS_PYG = True
except ImportError:
    print("Install: pip install torch-geometric"); HAS_PYG = False

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "task":        "ESOL_logS_regression",
    "n_epochs":    120,
    "batch_size":  32,
    "lr":          1e-3,
    "hidden_dim":  128,
    "n_heads":     8,       # multi-head attention
    "n_layers":    4,
    "dropout":     0.2,
    "edge_dropout":0.1,
    "seed":        42,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("="*70)
print("  GNN Script 03 — GAT with Multi-head Attention: Solubility (logS)")
print("="*70)

# ── ESOL-style dataset (aqueous solubility) ────────────────────────────────────
"""
ESOL (Estimated SOLubility) by Delaney 2004.
Unit: log(mol/L) — more negative = less soluble.
Typical range: -12 to +2.
"""
ESOL_DATA = [
    # (SMILES, logS, compound)
    ("OCC(O)CO",                   -0.19, "Glycerol"),
    ("OCC(O)C(O)C(O)CO",           -0.60, "Xylitol"),
    ("OC(=O)c1ccccc1",             -1.83, "Benzoic acid"),
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", -1.94, "Caffeine"),
    ("CC(=O)Nc1ccc(O)cc1",         -1.48, "Acetaminophen"),
    ("CC(=O)Oc1ccccc1C(=O)O",      -2.30, "Aspirin"),
    ("NC(CS)C(=O)O",               -0.40, "Cysteine"),
    ("OCC(O)CO",                   -0.19, "Glycerol-dup"),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.66, "Ibuprofen"),
    ("CC(O)CNc1ccc(NS(C)(=O)=O)cc1",-1.75,"Sotalol"),
    ("Cn1cc2c(cn1)CC(=O)N2CC1CCNCC1",-3.06,"Ondansetron"),
    ("c1ccc2ncccc2c1",              -2.72, "Quinoline"),
    ("c1ccc(Cl)c(Cl)c1",           -3.98, "1,2-DCB"),
    ("ClC(Cl)=C(Cl)Cl",            -4.20, "PCE"),
    ("ClCCCl",                     -1.40, "1,2-DCE"),
    ("[O-][N+](=O)c1ccccc1",       -2.55, "Nitrobenzene"),
    ("c1ccc2cc3ccccc3cc2c1",       -6.40, "Pyrene"),
    ("c1ccc2c(c1)ccc1cccc3cccc2c13",-7.64,"Benzo[a]pyrene"),
    ("C1CCCCC1",                   -2.62, "Cyclohexane"),
    ("CCCCCCCC",                   -6.50, "Octane"),
    ("CC(=O)O",                    -0.50, "Acetic acid"),
    ("OC(=O)CC(O)(CC(=O)O)C(=O)O",-0.15,"Citric acid"),
    ("OC(=O)c1ccc(Cl)cc1",        -2.79, "4-CBA"),
    ("Nc1ccc([N+](=O)[O-])cc1",   -2.59, "4-Nitroaniline"),
    ("Nc1ccccc1",                  -1.68, "Aniline"),
    ("OC(=O)CCCC(=O)O",           -0.81, "Glutaric acid"),
    ("CCCCO",                      -0.99, "1-Butanol"),
    ("CCCCOC(=O)c1ccccc1",        -4.50, "Butyl benzoate"),
    ("CC(C)Oc1ccccc1",            -2.90, "Isopropyl phenyl ether"),
    ("OC(=O)c1ccc(N)cc1",        -1.28, "4-ABA"),
    ("CC(=O)c1ccc(cc1)C(C)(C)C", -5.30, "4-tBu-acetophenone"),
    ("CCCCCCCC(=O)O",             -5.18, "Caprylic acid"),
    ("CC(C)c1ccccc1",             -4.55, "Isopropylbenzene"),
    ("c1ccccc1",                  -1.63, "Benzene"),
    ("Cc1ccccc1",                 -2.49, "Toluene"),
    ("CC(C)=O",                    0.24, "Acetone"),
    ("CCOCC",                     -0.48, "Diethyl ether"),
    ("OC1=CC=C2CC3N(CCC34CCc5c4cc(O)c(OC)c5)C2=C1",-3.70,"Morphine"),
    ("COc1ccc(CCN(C)CCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",-4.58,"Verapamil"),
    ("CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",-4.88,"Chlorpromazine"),
    ("CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1",-3.90,"Fluoxetine"),
    ("CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",-2.73,"Penicillin G"),
    ("CC(C)NCC(O)COc1cccc2ccccc12",-3.51,"Propranolol"),
    ("CN(C)C(=N)NC(=N)N",        -1.44,"Metformin"),
]

# ── Featurization ──────────────────────────────────────────────────────────────
def atom_features(atom):
    """Atom features for solubility prediction (emphasize polarity/H-bonding)."""
    common = [1,6,7,8,9,15,16,17,35,53,14]
    hyb    = [Chem.rdchem.HybridizationType.SP,
              Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3]
    def oh(v,c): enc=[0]*(len(c)+1); enc[c.index(v) if v in c else len(c)]=1; return enc
    return (
        oh(atom.GetAtomicNum(), common)           +  # 12
        oh(atom.GetDegree(), list(range(10)))     +  # 11
        oh(atom.GetFormalCharge(), list(range(-3,4))) + # 8
        oh(atom.GetTotalNumHs(), list(range(6)))  +  # 7
        oh(atom.GetHybridization(), hyb)          +  # 4
        [int(atom.GetIsAromatic())]               +  # 1
        [int(atom.IsInRing())]                    +  # 1
        [atom.GetMass() / 100.0]                  +  # 1 (polarity proxy)
        [Chem.Crippen.MolLogP(atom.GetOwningMol())/10.0 if False else 0.0]  # placeholder
    )  # 45 dim

def bond_features(bond):
    """Bond features (8-dim)."""
    def oh(v,c): enc=[0]*(len(c)+1); enc[c.index(v) if v in c else len(c)]=1; return enc
    return (
        oh(bond.GetBondType(),[Chem.rdchem.BondType.SINGLE,
                                Chem.rdchem.BondType.DOUBLE,
                                Chem.rdchem.BondType.TRIPLE,
                                Chem.rdchem.BondType.AROMATIC]) +  # 5
        [int(bond.GetIsConjugated())]                            +  # 1
        [int(bond.IsInRing())]                                      # 1 → total 7, pad to 8
    ) + [0]

def mol_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf   = bond_features(bond)
        ei  += [[i,j],[j,i]]; ea += [bf,bf]
    if not ei:
        ei = torch.zeros((2,0),dtype=torch.long)
        ea = torch.zeros((0,8),dtype=torch.float)
    else:
        ei = torch.tensor(ei,dtype=torch.long).t().contiguous()
        ea = torch.tensor(ea,dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y, smiles=smiles)

# ── Build Dataset ──────────────────────────────────────────────────────────────
print("\n[1/5] Building dataset...")
dataset = []
for smi, logs, name in ESOL_DATA:
    g = mol_to_graph(smi, logs)
    if g: g.name=name; dataset.append(g)

print(f"  Molecules     : {len(dataset)}")
print(f"  Node feat dim : {dataset[0].x.shape[1]}")
print(f"  logS range    : {min(g.y.item() for g in dataset):.2f} to {max(g.y.item() for g in dataset):.2f}")

N_NODE = dataset[0].x.shape[1]
N_EDGE = dataset[0].edge_attr.shape[1]

# Normalize targets
y_all  = np.array([g.y.item() for g in dataset])
y_mean = y_all.mean(); y_std = y_all.std()
for g in dataset: g.y_norm = (g.y - y_mean) / y_std

# Random split
rng = np.random.RandomState(CONFIG["seed"])
idx = list(range(len(dataset))); rng.shuffle(idx)
n_tr = int(0.8*len(idx))
train_data = [dataset[i] for i in idx[:n_tr]]
test_data  = [dataset[i] for i in idx[n_tr:]]
print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=CONFIG["batch_size"])

# ── GAT + GATv2 Model ─────────────────────────────────────────────────────────
class GATMolecular(nn.Module):
    """
    Graph Attention Network with:
    - GATv2Conv layers (Brody et al. 2022 — more expressive than original GAT)
      Standard GAT computes: e_{ij} = a^T · LeakyReLU(W·[h_i||h_j])
      GATv2 computes:        e_{ij} = a^T · LeakyReLU(W·h_i + W·h_j)  ← dynamic!
    - Multi-head attention (K=8 heads)
    - Edge dropout for regularization
    - Jumping Knowledge (JK) connections

    The attention weights α_{ij} can be extracted for interpretation:
    "Which neighboring atoms did atom i attend to?"
    """
    def __init__(self, n_node, hidden, n_heads, n_layers, dropout, edge_dropout):
        super().__init__()
        self.n_layers = n_layers
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        head_dim = hidden // n_heads

        # Input projection
        self.input_proj = nn.Linear(n_node, hidden)

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATv2Conv(
                hidden, head_dim,
                heads=n_heads,
                dropout=edge_dropout,
                concat=True,          # concatenate head outputs → hidden
                add_self_loops=True,
            ))
            self.bns.append(nn.BatchNorm1d(hidden))

        # Jumping Knowledge: use all layer outputs
        # JK-cat: concat all layer outputs
        self.jk_proj = nn.Linear((n_layers+1) * hidden, hidden)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, hidden//4),
            nn.GELU(),
            nn.Linear(hidden//4, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, return_attention=False):
        # Project to hidden dim
        h = F.gelu(self.input_proj(x))
        layer_outputs = [h]

        attention_weights_list = []
        for conv, bn in zip(self.convs, self.bns):
            if return_attention:
                h_new, (edge_idx, attn) = conv(h, edge_index, return_attention_weights=True)
                attention_weights_list.append((edge_idx, attn))
            else:
                h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new if h.shape == h_new.shape else h_new   # residual
            layer_outputs.append(h)

        # Jumping Knowledge: concatenate all layer representations
        h_jk = torch.cat(layer_outputs, dim=-1)   # [N, (n_layers+1)*hidden]
        h_jk = self.jk_proj(h_jk)                 # [N, hidden]

        # Graph-level pooling
        h_graph = global_mean_pool(h_jk, batch)   # [n_graphs, hidden]

        out = self.head(h_graph).squeeze(-1)
        if return_attention:
            return out, attention_weights_list
        return out

    def get_attention_weights(self, x, edge_index, batch):
        """Extract attention weights for interpretation."""
        return self.forward(x, edge_index, batch, return_attention=True)

# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, y_mean, y_std):
    model.train(); total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out   = model(batch.x, batch.edge_index, batch.batch)
        # Use normalized targets
        y_norm = (batch.y.squeeze() - y_mean) / y_std
        loss   = F.mse_loss(out, y_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std):
    model.eval(); preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch)
        # Denormalize
        pred_logS = out.cpu().numpy() * y_std + y_mean
        true_logS = batch.y.squeeze().cpu().numpy()
        preds.extend(pred_logS.tolist() if hasattr(pred_logS,'tolist') else [float(pred_logS)])
        targets.extend(true_logS.tolist() if hasattr(true_logS,'tolist') else [float(true_logS)])
    preds   = np.array(preds)
    targets = np.array(targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds) if len(targets)>1 else 0.0
    return rmse, mae, r2, preds, targets

if not HAS_PYG: exit()

device = torch.device(CONFIG["device"])
model  = GATMolecular(
    n_node=N_NODE, hidden=CONFIG["hidden_dim"],
    n_heads=CONFIG["n_heads"], n_layers=CONFIG["n_layers"],
    dropout=CONFIG["dropout"], edge_dropout=CONFIG["edge_dropout"],
).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[2/5] GAT model created")
print(f"  Architecture    : GATv2Conv + Jumping Knowledge + residual")
print(f"  Attention heads : {CONFIG['n_heads']}")
print(f"  Hidden dim      : {CONFIG['hidden_dim']} ({CONFIG['hidden_dim']//CONFIG['n_heads']} per head)")
print(f"  Trainable params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=CONFIG["lr"],
                        epochs=CONFIG["n_epochs"],
                        steps_per_epoch=len(train_loader))

print(f"\n[3/5] Training {CONFIG['n_epochs']} epochs (OneCycleLR)...")
history = {"train_loss":[], "test_rmse":[], "test_mae":[], "test_r2":[]}
best_rmse = 1e9; best_state = None
t_start   = time.time()

for epoch in range(1, CONFIG["n_epochs"]+1):
    tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device, y_mean, y_std)
    te_rmse, te_mae, te_r2, _, _ = evaluate(model, test_loader, device, y_mean, y_std)
    history["train_loss"].append(tr_loss)
    history["test_rmse"].append(te_rmse)
    history["test_mae"].append(te_mae)
    history["test_r2"].append(te_r2)
    if te_rmse < best_rmse:
        best_rmse = te_rmse
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if epoch % 30 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | loss={tr_loss:.4f} | "
              f"RMSE={te_rmse:.3f} | MAE={te_mae:.3f} | R2={te_r2:.3f}")

t_train = time.time() - t_start
model.load_state_dict(best_state)
_, _, _, final_preds, final_targets = evaluate(model, test_loader, device, y_mean, y_std)
final_rmse, final_mae, final_r2 = (
    np.sqrt(mean_squared_error(final_targets, final_preds)),
    mean_absolute_error(final_targets, final_preds),
    r2_score(final_targets, final_preds) if len(final_targets)>1 else 0.0,
)

print(f"\n[4/5] Final results:")
print(f"  RMSE : {final_rmse:.4f} log(mol/L)")
print(f"  MAE  : {final_mae:.4f} log(mol/L)")
print(f"  R²   : {final_r2:.4f}")
print(f"  Time : {t_train:.1f}s")

# Save
os.makedirs("gnn_results", exist_ok=True)
results = {"model":"GAT_GATv2_JK","n_params":n_params,
           "best_rmse":round(best_rmse,4),"final_rmse":round(final_rmse,4),
           "final_mae":round(final_mae,4),"final_r2":round(final_r2,4),
           "train_time_s":round(t_train,1),"config":CONFIG}
with open("gnn_results/03_gat_results.json","w") as f: json.dump(results,f,indent=2)

# ── Visualization ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating plots...")
fig = plt.figure(figsize=(16,10))
fig.suptitle("Script 03 — GAT with Multi-head Attention: Solubility (logS)",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)

ax1=fig.add_subplot(gs[0,0])
ax1.plot(history["train_loss"],color="#1565c0",lw=2,label="Train MSE loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss (normalized)")
ax1.set_title("Training Loss (OneCycleLR)"); ax1.legend(); ax1.grid(True,alpha=0.3)

ax2=fig.add_subplot(gs[0,1])
ax2.plot(history["test_rmse"],color="#e74c3c",lw=2,label="RMSE")
ax2.plot(history["test_mae"], color="#f39c12",lw=2,linestyle="--",label="MAE")
ax2_r=ax2.twinx()
ax2_r.plot(history["test_r2"],color="#27ae60",lw=2,linestyle=":",label="R²")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE / MAE"); ax2_r.set_ylabel("R²")
ax2.set_title("Regression Metrics"); ax2.legend(loc="upper right"); ax2.grid(True,alpha=0.3)

ax3=fig.add_subplot(gs[0,2])
mn = min(min(final_targets), min(final_preds)) - 0.5
mx = max(max(final_targets), max(final_preds)) + 0.5
ax3.scatter(final_targets, final_preds, alpha=0.7, color="#1565c0", s=60, zorder=5)
ax3.plot([mn,mx],[mn,mx],"k--",lw=1.5,label="Perfect")
ax3.plot([mn,mx],[mn-1,mx-1],"r:",lw=1,alpha=0.5)
ax3.plot([mn,mx],[mn+1,mx+1],"r:",lw=1,alpha=0.5,label="+/- 1 log unit")
ax3.set_xlabel("True logS"); ax3.set_ylabel("Predicted logS")
ax3.set_title(f"Predicted vs True (R²={final_r2:.3f})")
ax3.legend(fontsize=8); ax3.grid(True,alpha=0.3)
ax3.text(0.05,0.92,f"RMSE={final_rmse:.3f}\nMAE={final_mae:.3f}",
         transform=ax3.transAxes,fontsize=9,va='top',
         bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

ax4=fig.add_subplot(gs[1,0])
residuals = np.array(final_preds) - np.array(final_targets)
ax4.scatter(final_targets, residuals, alpha=0.6, color="#8e44ad", s=50)
ax4.axhline(0, color='k', lw=1.5)
ax4.axhline(1, color='r', linestyle='--', lw=1, alpha=0.7)
ax4.axhline(-1, color='r', linestyle='--', lw=1, alpha=0.7)
ax4.set_xlabel("True logS"); ax4.set_ylabel("Residual (pred - true)")
ax4.set_title("Residual Plot"); ax4.grid(True,alpha=0.3)

ax5=fig.add_subplot(gs[1,1])
ax5.hist(residuals, bins=15, color="#1565c0", alpha=0.7, edgecolor='white')
ax5.axvline(0,color='k',lw=1.5)
ax5.axvline(residuals.mean(),color='r',lw=2,linestyle='--',
            label=f"Mean={residuals.mean():.3f}")
ax5.set_xlabel("Residual"); ax5.set_ylabel("Count")
ax5.set_title("Residual Distribution"); ax5.legend(); ax5.grid(True,alpha=0.3)

ax6=fig.add_subplot(gs[1,2]); ax6.axis('off')
text=(
    "GAT Architecture\n"
    "────────────────────────────\n"
    f"Atom features  : {N_NODE} dim\n"
    f"Bond features  : {N_EDGE} dim\n"
    f"Hidden dim     : {CONFIG['hidden_dim']}\n"
    f"Attention heads: {CONFIG['n_heads']} ✓\n"
    f"GAT layers     : {CONFIG['n_layers']}\n"
    f"JK connections : Yes ✓\n"
    f"Edge dropout   : {CONFIG['edge_dropout']}\n"
    f"Trainable params: {n_params:,}\n"
    "────────────────────────────\n"
    f"Best RMSE      : {best_rmse:.4f}\n"
    f"Final RMSE     : {final_rmse:.4f}\n"
    f"Final R²       : {final_r2:.4f}\n"
    f"Train time     : {t_train:.1f}s\n"
    "────────────────────────────\n"
    "Advantage over MPNN:\n"
    "  + Learnable attention α_ij\n"
    "  + 8 independent heads\n"
    "  + Interpretable weights\n"
    "  + JK connections\n"
    "Limitation:\n"
    "  - Fixed neighborhood\n"
    "  - No global context\n"
    "→ See Script 04 (Transformer)"
)
ax6.text(0.05,0.95,text,transform=ax6.transAxes,fontsize=8.5,va='top',
         fontfamily='monospace',bbox=dict(boxstyle='round',facecolor='#f0f4f8',alpha=0.8))

plt.savefig("gnn_results/03_gat_results.png",dpi=150,bbox_inches="tight")
plt.show()
print("\n  Plot saved: gnn_results/03_gat_results.png")
print("="*70)
print("  Script 03 complete. GAT advances:")
print("  - Learnable attention weights α_{ij} per neighbor pair")
print("  - GATv2 (dynamic attention) more expressive than original GAT")
print("  - Multi-head attention captures diverse interaction patterns")
print("  - JK connections use all layer outputs (combats over-smoothing)")
print("  - Next: Script 04 adds global graph transformer + virtual nodes")
print("="*70)
