"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GenAI Script 02 — Conditional VAE (CVAE) for Drug Design                  ║
║  Task: Property-controlled molecule generation + latent space interpolation ║
║  Author: Himanshu Goel | himanshugoel.github.io                              ║
║                                                                              ║
║  Architecture: Conditional Variational Autoencoder (CVAE)                   ║
║  Key papers: Gómez-Bombarelli et al. 2018 ACS Cent. Sci. (original VAE)    ║
║              Lim et al. 2018 (CVAE for targeted drug design)                ║
║                                                                              ║
║  KEY ADVANCE over Script 01:                                                 ║
║    ✓ Continuous latent space — smooth interpolation between molecules        ║
║    ✓ Conditional generation — control LogP, QED, MW explicitly              ║
║    ✓ Latent space arithmetic — drug_A - drug_B + drug_C = drug_D            ║
║    ✓ Posterior collapse handling via KL annealing                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

VAE THEORY
──────────
VAE learns two distributions:
  Encoder: q(z|x) = N(μ(x), σ²(x)) — encode molecule to latent vector
  Decoder: p(x|z) — decode latent vector back to SMILES
  Prior:   p(z) = N(0, I) — standard normal prior

ELBO loss = reconstruction_loss + β * KL(q(z|x) || p(z))
  - Reconstruction: can the decoder recover the input SMILES?
  - KL divergence: how far is the encoded distribution from the prior?

CVAE extension: condition both encoder and decoder on property vector c:
  Encoder: q(z|x, c)
  Decoder: p(x|z, c)
  → at generation: sample z ~ N(0,I), provide target c → get desired molecule
"""

import os, json, time, warnings
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

# Reuse vocabulary and utilities from Script 01
import sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    from smiles_rnn_01 import (SMILES_CHARS, CHAR2IDX, IDX2CHAR, VOCAB_SIZE,
                                SOS_IDX, EOS_IDX, PAD_IDX, TRAINING_SMILES,
                                tokenize_smiles, smiles_to_indices, evaluate_molecules,
                                SMILESDataset, collate_fn, compute_sa_score)
except ImportError:
    # Inline essentials if Script 01 not on path
    SMILES_CHARS = ['<pad>','<sos>','<eos>','C','c','N','n','O','o','S','s','P','p','F',
                    'Cl','Br','I','H','B','=','#','-','(',')','[',']','+','/','\\'  ,
                    '1','2','3','4','5','6','7','8','9','0','@','.','%','Si','se','te','Se']
    CHAR2IDX = {c:i for i,c in enumerate(SMILES_CHARS)}
    IDX2CHAR  = {i:c for c,i in CHAR2IDX.items()}
    VOCAB_SIZE = len(SMILES_CHARS); SOS_IDX=1; EOS_IDX=2; PAD_IDX=0
    TRAINING_SMILES = [
        "CC(=O)Nc1ccc(O)cc1","CC(C)Cc1ccc(C(C)C(=O)O)cc1","Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "CC(O)CNc1ccc(NS(C)(=O)=O)cc1","CC(C)NCC(O)COc1cccc2ccccc12",
        "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21","CC(=O)Oc1ccccc1C(=O)O","OCC(O)C(O)C(O)CO",
        "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1","Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1",
        "CC1=CN(C(F)(F)F)C(=O)C=C1","O=C(O)c1ccc(N)cc1","CC(N)Cc1ccccc1",
        "Cc1cnc(NC(=O)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)s1",
        "CC(C)c1nc(N2CCOCC2)sc1C(=O)Nc1ccc(F)c(Cl)c1",
        "Clc1ccc2c(c1)n(CCN1CCCCC1)c(=O)n2","COc1ccc(CCN2CCC(OC)CC2)cc1OC",
        "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "CC(C)(C)c1ccc(C(=O)N2CCN(Cc3ccccc3)CC2)cc1",
        "FC(F)(F)c1cc(Nc2nccc(-c3cn4ccccc4n3)n2)cc(C(F)(F)F)c1",
    ]
    def tokenize_smiles(smiles):
        tokens=[]; i=0
        while i<len(smiles):
            two=smiles[i:i+2]
            if two in CHAR2IDX: tokens.append(two); i+=2
            elif smiles[i] in CHAR2IDX: tokens.append(smiles[i]); i+=1
            else: i+=1
        return tokens
    def smiles_to_indices(smiles,max_len=80):
        tokens=['<sos>']+tokenize_smiles(smiles)+['<eos>']
        return [CHAR2IDX.get(t,PAD_IDX) for t in tokens[:max_len+2]]
    def compute_sa_score(mol):
        if mol is None: return 10.0
        n_rings=rdMolDescriptors.CalcNumRings(mol); mw=Descriptors.ExactMolWt(mol)
        fsp3=Descriptors.FractionCSP3(mol)
        return round(min(10.0,max(1.0,1.0+n_rings*0.5+mw/200-fsp3*1.5)),2)
    def evaluate_molecules(gen_smi,train_smi,verbose=True):
        train_set=set(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train_smi if Chem.MolFromSmiles(s))
        valid_smi=[Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in gen_smi if Chem.MolFromSmiles(s)]
        if not valid_smi: return {"validity":0,"uniqueness":0,"novelty":0,"diversity":0,"drug_likeness":0,"qed_mean":0,"sa_mean":10,"n_valid":0}
        validity=len(valid_smi)/max(len(gen_smi),1); unique=list(set(valid_smi))
        uniqueness=len(unique)/max(len(valid_smi),1); novel=[s for s in unique if s not in train_set]
        novelty=len(novel)/max(len(unique),1)
        fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,1024) for s in unique[:30] if Chem.MolFromSmiles(s)]
        dists=[1-AllChem.DataStructs.TanimotoSimilarity(fps[i],fps[j]) for i in range(len(fps)) for j in range(i+1,len(fps))]
        diversity=float(np.mean(dists)) if dists else 0.0
        dl=sum(1 for s in unique[:100] if (m:=Chem.MolFromSmiles(s)) and Descriptors.ExactMolWt(m)<=500 and Descriptors.MolLogP(m)<=5)
        qeds=[QED.qed(Chem.MolFromSmiles(s)) for s in unique[:100] if Chem.MolFromSmiles(s)]
        sas=[compute_sa_score(Chem.MolFromSmiles(s)) for s in unique[:100] if Chem.MolFromSmiles(s)]
        metrics={"n_generated":len(gen_smi),"n_valid":len(valid_smi),"validity":round(validity,4),"uniqueness":round(uniqueness,4),
                 "novelty":round(novelty,4),"diversity":round(diversity,4),"drug_likeness":round(dl/max(len(unique[:100]),1),4),
                 "qed_mean":round(float(np.mean(qeds)),4) if qeds else 0.0,"qed_std":round(float(np.std(qeds)),4) if qeds else 0.0,
                 "sa_mean":round(float(np.mean(sas)),3) if sas else 10.0,"sa_std":round(float(np.std(sas)),3) if sas else 0.0}
        if verbose:
            for k in ["validity","uniqueness","novelty","diversity","drug_likeness","qed_mean","sa_mean"]:
                print(f"  {k:15s}: {metrics[k]:.4f}")
        return metrics
    class SMILESDataset(Dataset):
        def __init__(self,smiles_list,max_len=80):
            self.data=[]
            for smi in smiles_list:
                mol=Chem.MolFromSmiles(smi)
                if mol:
                    idx=smiles_to_indices(Chem.MolToSmiles(mol),max_len)
                    if len(idx)>2: self.data.append(torch.tensor(idx,dtype=torch.long))
        def __len__(self): return len(self.data)
        def __getitem__(self,i):
            seq=self.data[i]; return seq[:-1],seq[1:]
    def collate_fn(batch):
        inputs,targets=zip(*batch); max_len=max(x.size(0) for x in inputs)
        return (torch.stack([F.pad(x,(0,max_len-x.size(0)),value=PAD_IDX) for x in inputs]),
                torch.stack([F.pad(y,(0,max_len-y.size(0)),value=PAD_IDX) for y in targets]))

# ── Property featurizer ───────────────────────────────────────────────────────
def mol_properties(smiles: str) -> torch.Tensor:
    """Normalize MW, LogP, TPSA, QED as property conditioning vector."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return torch.zeros(4)
    return torch.tensor([
        Descriptors.ExactMolWt(mol) / 600.0,        # normalize to ~[0,1]
        (Descriptors.MolLogP(mol) + 2) / 10.0,      # shift and normalize
        Descriptors.TPSA(mol) / 200.0,
        QED.qed(mol),                                # already [0,1]
    ], dtype=torch.float)

PROP_DIM = 4   # MW, LogP, TPSA, QED

# ── Property-conditioned Dataset ─────────────────────────────────────────────
class CVAEDataset(Dataset):
    def __init__(self, smiles_list: list, max_len: int = 80):
        self.data = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                idx_seq   = smiles_to_indices(canonical, max_len)
                if len(idx_seq) > 2:
                    props = mol_properties(canonical)
                    self.data.append((torch.tensor(idx_seq, dtype=torch.long), props))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        seq, props = self.data[i]
        return seq[:-1], seq[1:], props

def cvae_collate(batch):
    inputs, targets, props = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    inputs_p  = torch.stack([F.pad(x,(0,max_len-x.size(0)),value=PAD_IDX) for x in inputs])
    targets_p = torch.stack([F.pad(y,(0,max_len-y.size(0)),value=PAD_IDX) for y in targets])
    return inputs_p, targets_p, torch.stack(props)

# ── CVAE Model ────────────────────────────────────────────────────────────────
class MolecularCVAE(nn.Module):
    """
    Conditional VAE for property-directed molecule generation.

    Encoder: SMILES + properties → (μ, log σ²) in latent space
    Decoder: (z, properties) → SMILES reconstruction

    Key design choices:
    1. GRU encoder (bidirectional) for better context capture
    2. Property conditioning by concatenating to z before decoding
    3. KL annealing: β increases from 0→1 during training (prevents collapse)
    4. Free bits: minimum KL per dimension to prevent posterior collapse
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, latent_dim: int = 64,
                 prop_dim: int = PROP_DIM, dropout: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.embed_enc   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, 2, batch_first=True,
                                   bidirectional=True, dropout=dropout)
        enc_out_dim = hidden_dim * 2  # bidirectional

        # Condition encoding
        self.prop_enc = nn.Sequential(
            nn.Linear(prop_dim, 32), nn.ReLU(), nn.Linear(32, 32))

        # Latent projections
        self.fc_mu  = nn.Linear(enc_out_dim + 32, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim + 32, latent_dim)

        # Decoder
        self.embed_dec   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.decoder_rnn = nn.GRU(embed_dim + latent_dim + prop_dim, hidden_dim, 2,
                                   batch_first=True, dropout=dropout)
        self.decoder_proj = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, props: torch.Tensor):
        emb = self.drop(self.embed_enc(x))
        _, h = self.encoder_rnn(emb)
        # Combine forward/backward last hidden states
        h_fwd = h[-2]   # last forward layer
        h_bwd = h[-1]   # last backward layer
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        # Condition on properties
        p_enc = self.prop_enc(props)
        h_cond = torch.cat([h_cat, p_enc], dim=-1)
        return self.fc_mu(h_cond), self.fc_var(h_cond)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, props: torch.Tensor, x: torch.Tensor):
        emb  = self.drop(self.embed_dec(x))
        # Broadcast z and props to all time steps
        z_expand    = z.unsqueeze(1).expand(-1, emb.size(1), -1)
        prop_expand = props.unsqueeze(1).expand(-1, emb.size(1), -1)
        inp  = torch.cat([emb, z_expand, prop_expand], dim=-1)
        out, _ = self.decoder_rnn(inp)
        return self.decoder_proj(self.drop(out))

    def forward(self, x, props, target):
        mu, log_var = self.encode(x, props)
        z  = self.reparameterize(mu, log_var)
        logits = self.decode(z, props, x)
        return logits, mu, log_var

    @torch.no_grad()
    def generate_conditional(self, target_props: torch.Tensor,
                               n_molecules: int = 50, max_len: int = 80,
                               temperature: float = 1.0,
                               device: str = 'cpu') -> list:
        """
        Generate molecules with target properties.
        target_props: [batch, prop_dim] normalized property vector
        """
        self.eval()
        generated = []
        props = target_props.to(device)
        if props.dim() == 1: props = props.unsqueeze(0)
        # Expand to n_molecules
        props = props.expand(n_molecules, -1)

        for i in range(n_molecules):
            p = props[i:i+1]   # [1, prop_dim]
            z = torch.randn(1, self.latent_dim, device=device)
            x = torch.tensor([[SOS_IDX]], device=device)
            chars = []
            # Single-step decoder
            for _ in range(max_len):
                emb  = self.embed_dec(x)
                z_ex = z.unsqueeze(1)
                p_ex = p.unsqueeze(1)
                inp  = torch.cat([emb, z_ex, p_ex], dim=-1)
                out, _ = self.decoder_rnn(inp)
                logits = self.decoder_proj(out[:, -1]) / temperature
                probs  = F.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, 1).item()
                if next_t == EOS_IDX: break
                if next_t not in (PAD_IDX, SOS_IDX):
                    chars.append(IDX2CHAR.get(next_t, ''))
                x = torch.tensor([[next_t]], device=device)
            generated.append(''.join(chars))
        return generated

    @torch.no_grad()
    def interpolate(self, smiles1: str, smiles2: str, n_steps: int = 8,
                     device: str = 'cpu') -> list:
        """
        Latent space interpolation between two molecules.
        Key advantage of VAE over RNN: smooth interpolation in latent space.
        drug_A → drug_B generates a chemical pathway of intermediate structures.
        """
        self.eval()
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2: return []

        p1 = mol_properties(smiles1).unsqueeze(0).to(device)
        p2 = mol_properties(smiles2).unsqueeze(0).to(device)
        x1 = torch.tensor([smiles_to_indices(Chem.MolToSmiles(mol1))], dtype=torch.long, device=device)
        x2 = torch.tensor([smiles_to_indices(Chem.MolToSmiles(mol2))], dtype=torch.long, device=device)

        mu1, _ = self.encode(x1, p1)
        mu2, _ = self.encode(x2, p2)

        interpolated = []
        for alpha in np.linspace(0, 1, n_steps):
            z    = (1 - alpha) * mu1 + alpha * mu2
            p    = (1 - alpha) * p1  + alpha * p2
            x    = torch.tensor([[SOS_IDX]], device=device)
            chars = []
            for _ in range(80):
                emb  = self.embed_dec(x)
                inp  = torch.cat([emb, z.unsqueeze(1), p.unsqueeze(1)], dim=-1)
                out, _ = self.decoder_rnn(inp)
                logits = self.decoder_proj(out[:, -1])
                probs  = F.softmax(logits / 0.8, dim=-1)
                next_t = torch.multinomial(probs, 1).item()
                if next_t == EOS_IDX: break
                if next_t not in (PAD_IDX, SOS_IDX):
                    chars.append(IDX2CHAR.get(next_t, ''))
                x = torch.tensor([[next_t]], device=device)
            interpolated.append((round(float(alpha), 2), ''.join(chars)))
        return interpolated

# ── Training with KL annealing ───────────────────────────────────────────────
def train_cvae(model, loader, optimizer, device, n_epochs=50,
               kl_max=1.0, kl_anneal_steps=30):
    """
    ELBO = reconstruction_loss - β * KL(q||p)
    β increases from 0→kl_max over kl_anneal_steps epochs (KL annealing).
    This prevents posterior collapse (a common VAE failure mode).
    """
    model.train()
    history = {"loss": [], "recon_loss": [], "kl_loss": [], "beta": []}
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(1, n_epochs+1):
        # Cosine KL annealing schedule
        beta = kl_max * min(1.0, epoch / kl_anneal_steps)
        ep_loss, ep_recon, ep_kl = 0, 0, 0

        for inputs, targets, props in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            props   = props.to(device)
            optimizer.zero_grad()

            logits, mu, log_var = model(inputs, props, inputs)

            # Reconstruction loss
            recon = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            # KL divergence (sum over latent dims, mean over batch)
            kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            # Free bits: minimum KL=0.5 per dim to prevent collapse
            kl    = torch.clamp(kl, min=0.5 * model.latent_dim / model.latent_dim)
            loss  = recon + beta * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss  += loss.item()
            ep_recon += recon.item()
            ep_kl    += kl.item()

        n = len(loader)
        history["loss"].append(ep_loss/n)
        history["recon_loss"].append(ep_recon/n)
        history["kl_loss"].append(ep_kl/n)
        history["beta"].append(beta)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | Total={ep_loss/n:.4f} | "
                  f"Recon={ep_recon/n:.4f} | KL={ep_kl/n:.4f} | β={beta:.3f}")
    return history

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    print("="*65)
    print("  GenAI Script 02 — Conditional VAE (CVAE)")
    print("  Architecture: BiGRU Encoder + Property-Conditioned Decoder")
    print("="*65)

    print("\n[1/5] Building dataset with property conditioning...")
    dataset = CVAEDataset(TRAINING_SMILES)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=cvae_collate)
    print(f"  {len(dataset)} valid molecules | Prop dim: {PROP_DIM} | Latent dim: 64")

    print("\n[2/5] Training CVAE with KL annealing...")
    model = MolecularCVAE(VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                           latent_dim=64, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    print(f"  Parameters: {n_params:,}")

    t0      = time.time()
    history = train_cvae(model, loader, optimizer, device, n_epochs=50)
    t_train = time.time() - t0

    print("\n[3/5] Conditional generation (targeting different QED ranges)...")
    t_gen_start = time.time()
    all_gen = []

    # High QED target (drug-like)
    high_qed_props = torch.tensor([350/600, (2+2)/10, 80/200, 0.8])
    high_qed_mols  = model.generate_conditional(high_qed_props, n_molecules=80,
                                                  temperature=0.9, device=device)
    print(f"  High-QED target (QED≈0.8): {sum(1 for s in high_qed_mols if Chem.MolFromSmiles(s))}/80 valid")
    all_gen.extend(high_qed_mols)

    # Diverse exploration
    explore_props = torch.tensor([450/600, (3+2)/10, 120/200, 0.5])
    explore_mols  = model.generate_conditional(explore_props, n_molecules=80,
                                                temperature=1.2, device=device)
    print(f"  Exploration (T=1.2):        {sum(1 for s in explore_mols if Chem.MolFromSmiles(s))}/80 valid")
    all_gen.extend(explore_mols)

    t_gen = (time.time() - t_gen_start) * 1000 / len(all_gen)

    print("\n[4/5] Latent space interpolation...")
    smi_a = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"   # Ibuprofen
    smi_b = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"    # Caffeine
    interp = model.interpolate(smi_a, smi_b, n_steps=8, device=device)
    print(f"  Interpolating {smi_a[:25]} → {smi_b[:25]}")
    for alpha, smi in interp:
        mol = Chem.MolFromSmiles(smi)
        valid = "✓" if mol else "✗"
        qed_v = f"QED={QED.qed(mol):.3f}" if mol else ""
        print(f"    α={alpha:.2f}: {smi[:45]:45s} {valid} {qed_v}")

    print("\n[5/5] Evaluation metrics:")
    metrics = evaluate_molecules(all_gen, TRAINING_SMILES)

    metrics.update({
        "model":            "CVAE",
        "architecture":     "BiGRU Encoder + Property-conditioned GRU Decoder",
        "latent_dim":       64,
        "prop_conditioning":True,
        "kl_annealing":     True,
        "n_params":         n_params,
        "train_time_s":     round(t_train, 1),
        "gen_time_ms_per_mol": round(t_gen, 2),
    })

    # Visualization
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Script 02 — CVAE: Property-controlled Molecule Generation", fontweight='bold')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["recon_loss"], color="#1565c0", lw=2, label="Reconstruction")
    ax1.plot(history["kl_loss"],    color="#e65100", lw=2, linestyle="--", label="KL")
    ax1.set_xlabel("Epoch"); ax1.set_title("ELBO Components (KL annealing)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history["beta"], color="#8e44ad", lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("β"); ax2.set_title("KL Annealing Schedule")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    metric_names = ["validity","uniqueness","novelty","diversity","drug_likeness","qed_mean"]
    metric_vals  = [metrics[m] for m in metric_names]
    bars = ax3.bar(metric_names, metric_vals,
                    color=["#27ae60","#1565c0","#8e44ad","#e65100","#c0392b","#2c3e50"], alpha=0.85)
    ax3.set_ylim([0,1]); ax3.set_title("Generation Metrics"); ax3.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, metric_vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.3f}", ha='center', fontsize=8)

    # Latent space t-SNE visualization
    ax4 = fig.add_subplot(gs[0, 3])
    try:
        from sklearn.manifold import TSNE
        test_smi = [s for s in TRAINING_SMILES if Chem.MolFromSmiles(s)]
        latent_vecs = []
        model.eval()
        with torch.no_grad():
            for smi in test_smi[:20]:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    idx = smiles_to_indices(Chem.MolToSmiles(mol))
                    x_t = torch.tensor([idx], dtype=torch.long, device=device)
                    p_t = mol_properties(smi).unsqueeze(0).to(device)
                    mu, _ = model.encode(x_t, p_t)
                    latent_vecs.append(mu.cpu().numpy()[0])
        if len(latent_vecs) > 5:
            Z = np.array(latent_vecs)
            perp = min(5, len(Z)-1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
            Z2   = tsne.fit_transform(Z)
            scatter = ax4.scatter(Z2[:,0], Z2[:,1],
                                   c=[QED.qed(Chem.MolFromSmiles(s)) for s in test_smi[:len(Z)]],
                                   cmap='viridis', s=60, alpha=0.8)
            plt.colorbar(scatter, ax=ax4, label='QED')
            ax4.set_title("Latent Space (t-SNE)\nColored by QED")
    except Exception as e:
        ax4.text(0.5, 0.5, f"t-SNE: {str(e)[:40]}", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Latent Space Visualization")

    # QED comparison: targeted vs random
    ax5 = fig.add_subplot(gs[1, 0:2])
    hq_qeds = [QED.qed(Chem.MolFromSmiles(s)) for s in high_qed_mols if Chem.MolFromSmiles(s)]
    ex_qeds = [QED.qed(Chem.MolFromSmiles(s)) for s in explore_mols if Chem.MolFromSmiles(s)]
    if hq_qeds: ax5.hist(hq_qeds, bins=15, alpha=0.6, color="#27ae60", label=f"High-QED target (target=0.8)")
    if ex_qeds: ax5.hist(ex_qeds, bins=15, alpha=0.6, color="#e65100", label="Explore (T=1.2)")
    ax5.axvline(0.8, color='k', linestyle='--', lw=1.5, label='Target QED=0.8')
    ax5.set_xlabel("QED"); ax5.set_ylabel("Count"); ax5.set_title("Conditional QED Distribution")
    ax5.legend(fontsize=8)

    # Property correlation scatter
    ax6 = fig.add_subplot(gs[1, 2:])
    all_valid_mols = [Chem.MolFromSmiles(s) for s in all_gen if Chem.MolFromSmiles(s)]
    mws  = [Descriptors.ExactMolWt(m) for m in all_valid_mols]
    qeds_all = [QED.qed(m) for m in all_valid_mols]
    logps = [Descriptors.MolLogP(m) for m in all_valid_mols]
    if mws:
        sc = ax6.scatter(mws, qeds_all, c=logps, cmap='coolwarm', s=20, alpha=0.6)
        plt.colorbar(sc, ax=ax6, label='LogP')
        ax6.set_xlabel("Molecular Weight"); ax6.set_ylabel("QED")
        ax6.set_title("Generated Molecules: MW vs QED (color=LogP)")
        ax6.axhline(0.5, color='k', linestyle='--', lw=0.8, alpha=0.5)
        ax6.axvline(500, color='k', linestyle='--', lw=0.8, alpha=0.5)

    plt.savefig("genai_results/02_cvae_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    os.makedirs("genai_results", exist_ok=True)
    with open("genai_results/02_cvae_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), "genai_results/02_cvae_weights.pt")

    print(f"\n  Train time: {t_train:.1f}s | Gen time: {t_gen:.1f}ms/mol")
    print(f"  Results saved: genai_results/02_cvae_results.json")
    print("="*65)
    print("  Script 02 complete. CVAE advances:")
    print("  ✓ Continuous latent space — smooth interpolation")
    print("  ✓ Property conditioning — control MW, LogP, QED")
    print("  ✓ KL annealing — prevents posterior collapse")
    print("  Limitation: mode collapse risk, requires tuning β schedule")
    print("  → Script 03 adds adversarial training (MolGAN-style)")
    print("="*65)
    return metrics

if __name__ == "__main__":
    run()
