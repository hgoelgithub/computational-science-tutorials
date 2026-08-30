"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GenAI Script 01 — Character-level SMILES RNN (Baseline)                    ║
║  Task: De novo drug-like molecule generation                                 ║
║  Author: Himanshu Goel | hgoelgithub.github.io                              ║
║                                                                              ║
║  Architecture: LSTM language model on SMILES strings                         ║
║  Seminal paper: Segler et al. 2018, ACS Central Science                      ║
║  "Generating Focused Molecule Libraries for Drug Discovery"                  ║
║                                                                              ║
║  This is the FOUNDING ARCHITECTURE of modern drug design generative AI.      ║
║  REINVENT (AstraZeneca) started exactly like this in 2017.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW IT WORKS
────────────
The RNN learns the probability distribution over SMILES characters:
  P(SMILES) = ∏ P(char_t | char_1, ..., char_{t-1})

It reads training SMILES one character at a time and learns to predict
the next character. At generation time, it samples from this learned
distribution autoregressively.

This is analogous to GPT — but for molecular strings, not English text.

KEY METRICS (standard across all 5 scripts):
  Validity:   % of generated SMILES that parse to valid RDKit molecules
  Uniqueness: % of valid molecules that are unique (deduplicated)
  Novelty:    % of unique valid molecules NOT in the training set
  Diversity:  mean pairwise Tanimoto distance (higher = more diverse)
  Drug-likeness: % satisfying Lipinski Ro5
  QED:        Quantitative Estimate of Drug-likeness (Bickerton 2012)
  SA score:   Synthetic Accessibility (Ertl & Schuffenhauer 2009)
"""

import os, json, time, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED

# ── Vocabulary ─────────────────────────────────────────────────────────────────
# Standard SMILES vocabulary for drug-like molecules
SMILES_CHARS = [
    '<pad>', '<sos>', '<eos>',
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'Cl', 'Br', 'I',
    'H', 'B', '=', '#', '-', '(', ')', '[', ']', '+', '/', '\\',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '@', '.', '%',
    'Si', 'se', 'te', 'Se',
]
CHAR2IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
IDX2CHAR  = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(SMILES_CHARS)

SOS_IDX = CHAR2IDX['<sos>']
EOS_IDX = CHAR2IDX['<eos>']
PAD_IDX = CHAR2IDX['<pad>']

# ── Training Data (drug-like molecules from ChEMBL/ZINC) ──────────────────────
TRAINING_SMILES = [
    # FDA-approved drugs and drug-like molecules
    "CC(=O)Nc1ccc(O)cc1",            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",   "CC(O)CNc1ccc(NS(C)(=O)=O)cc1",
    "CC(C)NCC(O)COc1cccc2ccccc12",   "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
    "CC1=C(C(=O)Nc2ccccc2)C(c2ccccc2)N(C(C)=O)C1", "c1ccc2ncccc2c1",
    "CC(=O)Oc1ccccc1C(=O)O",         "OCC(O)C(O)C(O)CO",
    "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1",
    "COc1ccc2c(c1)C(=O)c1cc(OC)c(OC)cc1C2=O",
    "CC(C)(C)c1ccc(O)c(O)c1",        "OC1=CC=C2CC3N(CCC34CCc5c4cc(O)c(OC)c5)C2=C1",
    "COc1cc2c(cc1OC)CC1CC3N(CCc4cc(OC)c(OC)cc4C13)CC21",
    "CC1=CN(C(F)(F)F)C(=O)C=C1",    "Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1",
    "O=C(O)c1ccc(N)cc1",             "CC(N)Cc1ccccc1",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",   "Cc1cnc(NC(=O)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)s1",
    "O=c1[nH]c2ccccc2n1Cc1ccccc1",  "CC(C)c1nc(N2CCOCC2)sc1C(=O)Nc1ccc(F)c(Cl)c1",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "COc1ccc(C2C(C(=O)OC(C)C)=C(C)Nc3ccccc32)cc1OC",
    "Clc1ccc2c(c1)n(CCN1CCCCC1)c(=O)n2",
    "CC1(C)CCC(=C2C=C(c3ccc(F)cc3)NC2=O)CC1",
    "COc1cccc(NC(=O)c2ccc(C#N)cc2)c1",
    "FC(F)(F)c1cc(Nc2nccc(-c3cn4ccccc4n3)n2)cc(C(F)(F)F)c1",
    "CC(C)c1ccc(C(=O)Nc2ccc(Cl)c(Cl)c2)cc1",
    "Cc1cc(NC(=O)c2ccc(F)cc2)no1",  "O=C(Cc1ccccc1)NCc1ccccc1",
    "CC(C)(C)OC(=O)Nc1ccc(B(O)O)cc1",
    "Nc1nc(=O)c2ncn(C3OC(CO)C(O)C3O)c2[nH]1",
    "CC(=O)N1CCN(c2ccc(OC3CCCC3)cc2)CC1",
    "COc1ccc(CCN2CCC(OC)CC2)cc1OC",
    "c1cnc(Nc2ccccc2)nc1",           "O=C(O)c1cccc(O)c1",
    "Nc1ccc2nc(=O)[nH]cc2c1",        "CC(=O)c1ccc(NC(=O)c2ccco2)cc1",
    "CCOC(=O)c1ccc(NC(C)=O)cc1",    "c1ccc(CNc2ncnc3sccc23)cc1",
    "Cc1ccc(NC(=O)Nc2ccc(Cl)c(Cl)c2)cc1",
    "CC(C)(C)c1ccc(C(=O)N2CCN(Cc3ccccc3)CC2)cc1",
]

# ── SMILES Tokenizer ───────────────────────────────────────────────────────────
def tokenize_smiles(smiles: str) -> list:
    """Tokenize SMILES into multi-character and single-character tokens."""
    tokens = []
    i = 0
    while i < len(smiles):
        # Try 2-char tokens first (Cl, Br, Si, etc.)
        two_char = smiles[i:i+2]
        if two_char in CHAR2IDX:
            tokens.append(two_char)
            i += 2
        elif smiles[i] in CHAR2IDX:
            tokens.append(smiles[i])
            i += 1
        else:
            i += 1  # skip unknown characters
    return tokens

def smiles_to_indices(smiles: str, max_len: int = 80) -> list:
    tokens = ['<sos>'] + tokenize_smiles(smiles) + ['<eos>']
    indices = [CHAR2IDX.get(t, PAD_IDX) for t in tokens[:max_len+2]]
    return indices

# ── Dataset ────────────────────────────────────────────────────────────────────
class SMILESDataset(Dataset):
    def __init__(self, smiles_list: list, max_len: int = 80):
        self.data = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                idx_seq   = smiles_to_indices(canonical, max_len)
                if len(idx_seq) > 2:
                    self.data.append(torch.tensor(idx_seq, dtype=torch.long))

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        seq = self.data[i]
        return seq[:-1], seq[1:]   # input, target (shifted by 1)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    inputs_pad  = torch.stack([F.pad(x, (0, max_len-x.size(0)), value=PAD_IDX) for x in inputs])
    targets_pad = torch.stack([F.pad(y, (0, max_len-y.size(0)), value=PAD_IDX) for y in targets])
    return inputs_pad, targets_pad

# ── LSTM RNN Model ─────────────────────────────────────────────────────────────
class SMILESRNN(nn.Module):
    """
    Character-level LSTM for SMILES generation.

    Architecture: Segler et al. 2018 (ACS Cent. Sci.)
    - Embedding(vocab) → LSTM(hidden) × n_layers → Linear(vocab)
    - Teacher forcing during training
    - Temperature-controlled sampling during generation

    This exact architecture underpins REINVENT 1.0-3.0 (AstraZeneca)
    and was the de facto standard 2017-2021.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 512, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=dropout)
        self.head  = nn.Linear(hidden_dim, vocab_size)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        emb = self.drop(self.embed(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(self.drop(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return h, c

    @torch.no_grad()
    def generate(self, n_molecules: int = 100, max_len: int = 80,
                  temperature: float = 1.0, device: str = 'cpu') -> list:
        """
        Autoregressive SMILES generation with temperature sampling.

        Temperature controls creativity:
          T < 1.0 → conservative, high-validity (more drug-like)
          T = 1.0 → balanced
          T > 1.0 → creative, more diverse (more invalid)
        """
        self.eval()
        generated = []

        for _ in range(n_molecules):
            hidden = self.init_hidden(1, device)
            x = torch.tensor([[SOS_IDX]], device=device)
            chars = []

            for _ in range(max_len):
                logits, hidden = self(x, hidden)
                logits  = logits[:, -1, :] / temperature
                probs   = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

                if next_tok == EOS_IDX:
                    break
                if next_tok not in (PAD_IDX, SOS_IDX):
                    chars.append(IDX2CHAR.get(next_tok, ''))

                x = torch.tensor([[next_tok]], device=device)

            generated.append(''.join(chars))

        return generated

# ── Molecular Evaluation Metrics ──────────────────────────────────────────────
def compute_sa_score(mol) -> float:
    """
    Synthetic Accessibility (SA) score (Ertl & Schuffenhauer 2009).
    Range 1-10: 1 = very easy to synthesize, 10 = very hard.
    """
    # Simplified SA using fragment complexity as proxy
    if mol is None: return 10.0
    n_rings      = rdMolDescriptors.CalcNumRings(mol)
    n_stereo     = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_fused      = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    mw           = Descriptors.ExactMolWt(mol)
    fsp3         = Descriptors.FractionCSP3(mol)
    # Higher complexity = higher SA score
    complexity   = (n_rings * 0.5 + n_stereo * 0.8 + n_fused * 1.5 + mw/200)
    sa           = min(10.0, max(1.0, 1.0 + complexity * 0.4 - fsp3 * 1.5))
    return round(sa, 2)

def evaluate_molecules(generated_smiles: list, train_smiles: list, verbose: bool = True) -> dict:
    """
    Industry-standard evaluation suite for molecular generative models.
    Based on MOSES benchmark (Polykovskiy et al. 2020) and GuacaMol.
    """
    train_set = set(Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    for s in train_smiles if Chem.MolFromSmiles(s))

    valid_mols, valid_smi = [], []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            valid_mols.append(mol)
            valid_smi.append(canonical)

    if not valid_smi:
        return {"validity":0,"uniqueness":0,"novelty":0,"diversity":0,
                "drug_likeness":0,"qed_mean":0,"sa_mean":10,"n_valid":0}

    validity    = len(valid_smi) / max(len(generated_smiles), 1)
    unique_smi  = list(set(valid_smi))
    uniqueness  = len(unique_smi) / max(len(valid_smi), 1)
    novel_smi   = [s for s in unique_smi if s not in train_set]
    novelty     = len(novel_smi) / max(len(unique_smi), 1)

    # Internal diversity (mean pairwise Tanimoto distance)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024)
           for s in unique_smi[:50] if Chem.MolFromSmiles(s)]
    diversity = 0.0
    if len(fps) > 1:
        dists = []
        for i in range(min(len(fps), 30)):
            for j in range(i+1, min(len(fps), 30)):
                sim = AllChem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(1 - sim)
        diversity = float(np.mean(dists)) if dists else 0.0

    # Drug-likeness and QED
    dl_count, qed_scores, sa_scores = 0, [], []
    for mol in [Chem.MolFromSmiles(s) for s in unique_smi[:100] if Chem.MolFromSmiles(s)]:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            dl_count += 1
        try:
            qed_scores.append(QED.qed(mol))
        except: pass
        sa_scores.append(compute_sa_score(mol))

    drug_likeness = dl_count / max(len(unique_smi[:100]), 1)

    metrics = {
        "n_generated":  len(generated_smiles),
        "n_valid":      len(valid_smi),
        "validity":     round(validity, 4),
        "uniqueness":   round(uniqueness, 4),
        "novelty":      round(novelty, 4),
        "diversity":    round(diversity, 4),
        "drug_likeness":round(drug_likeness, 4),
        "qed_mean":     round(float(np.mean(qed_scores)), 4) if qed_scores else 0.0,
        "qed_std":      round(float(np.std(qed_scores)), 4) if qed_scores else 0.0,
        "sa_mean":      round(float(np.mean(sa_scores)), 3) if sa_scores else 10.0,
        "sa_std":       round(float(np.std(sa_scores)), 3) if sa_scores else 0.0,
    }

    if verbose:
        print(f"  Validity:      {metrics['validity']:.3f} ({metrics['n_valid']}/{metrics['n_generated']})")
        print(f"  Uniqueness:    {metrics['uniqueness']:.3f}")
        print(f"  Novelty:       {metrics['novelty']:.3f}")
        print(f"  Diversity:     {metrics['diversity']:.3f}")
        print(f"  Drug-likeness: {metrics['drug_likeness']:.3f}")
        print(f"  QED (mean):    {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
        print(f"  SA  (mean):    {metrics['sa_mean']:.3f} ± {metrics['sa_std']:.3f}")

    return metrics

# ── Training Loop ─────────────────────────────────────────────────────────────
def train(model, loader, optimizer, device, n_epochs=30):
    model.train()
    history = {"loss": []}
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(1, n_epochs+1):
        epoch_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            # Reshape for cross-entropy: (batch*seq, vocab)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history["loss"].append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f} | "
                  f"Perplexity: {np.exp(avg_loss):.2f}")

    return history

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    print("="*65)
    print("  GenAI Script 01 — Character-level SMILES RNN")
    print("  Architecture: LSTM (Segler 2018 / REINVENT 1.0)")
    print("="*65)

    # Dataset
    print("\n[1/4] Building dataset...")
    dataset = SMILESDataset(TRAINING_SMILES)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    print(f"  {len(dataset)} valid training molecules | Vocab: {VOCAB_SIZE}")

    # Model
    print("\n[2/4] Training RNN...")
    model = SMILESRNN(VOCAB_SIZE, embed_dim=128, hidden_dim=512,
                       n_layers=3, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"  Parameters: {n_params:,}")

    t0      = time.time()
    history = train(model, loader, optimizer, device, n_epochs=40)
    t_train = time.time() - t0

    # Generate
    print("\n[3/4] Generating molecules...")
    t_gen_start = time.time()
    generated   = model.generate(n_molecules=200, temperature=0.9, device=device)
    t_gen       = (time.time() - t_gen_start) * 1000 / 200   # ms per molecule

    # Evaluate
    print("\n[4/4] Evaluation metrics (MOSES standard):")
    metrics = evaluate_molecules(generated, TRAINING_SMILES)

    # Show sample molecules
    valid_gen = [s for s in generated if Chem.MolFromSmiles(s)][:5]
    print(f"\n  Sample valid molecules:")
    for i, smi in enumerate(valid_gen):
        mol  = Chem.MolFromSmiles(smi)
        mw   = round(Descriptors.ExactMolWt(mol), 1)
        logp = round(Descriptors.MolLogP(mol), 2)
        qed  = round(QED.qed(mol), 3)
        print(f"    {i+1}. {smi[:50]:50s} MW={mw} LogP={logp} QED={qed}")

    metrics.update({
        "model": "SMILES_RNN",
        "architecture": "LSTM 3-layer (Segler 2018)",
        "n_params": n_params,
        "train_time_s": round(t_train, 1),
        "gen_time_ms_per_mol": round(t_gen, 2),
        "temperature": 0.9,
    })

    # Save
    os.makedirs("genai_results", exist_ok=True)
    with open("genai_results/01_rnn_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), "genai_results/01_rnn_weights.pt")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Script 01 — SMILES RNN: Training & Generated Molecules", fontweight='bold')

    axes[0].plot(history["loss"], color="#1565c0", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy Loss")
    axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)

    metric_names = ["validity","uniqueness","novelty","diversity","drug_likeness","qed_mean"]
    metric_vals  = [metrics[m] for m in metric_names]
    bars = axes[1].bar(metric_names, metric_vals,
                        color=["#27ae60","#1565c0","#8e44ad","#e65100","#c0392b","#2c3e50"],
                        alpha=0.85)
    axes[1].set_ylim([0,1]); axes[1].set_title("Generation Metrics")
    axes[1].tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, metric_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                      f"{val:.3f}", ha='center', fontsize=8)

    # QED distribution
    qeds = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try: qeds.append(QED.qed(mol))
            except: pass
    if qeds:
        axes[2].hist(qeds, bins=20, color="#1565c0", alpha=0.7, edgecolor='white')
        axes[2].axvline(np.mean(qeds), color='red', lw=2, linestyle='--',
                         label=f"Mean={np.mean(qeds):.3f}")
        axes[2].set_xlabel("QED Score"); axes[2].set_ylabel("Count")
        axes[2].set_title("QED Distribution (generated)"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("genai_results/01_rnn_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n  Train time: {t_train:.1f}s | Gen time: {t_gen:.1f}ms/mol")
    print(f"  Results saved: genai_results/01_rnn_results.json")
    print("="*65)
    print("  Script 01 complete. Baseline RNN established.")
    print("  Limitation: no property control, no latent space interpolation")
    print("  → Script 02 adds VAE for continuous latent space manipulation")
    print("="*65)
    return metrics

if __name__ == "__main__":
    run()
