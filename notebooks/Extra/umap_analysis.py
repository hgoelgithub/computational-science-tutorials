#!/usr/bin/env python3
"""
Advanced Chemical Space Analysis
A comprehensive tutorial on chemical space visualization using dimensionality reduction
Note: UMAP requires Python < 3.13. Using PCA and t-SNE for demonstration.
"""

import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("Libraries imported successfully!")

# Comprehensive molecular dataset
molecular_data = {
    'NSAIDs': [
        ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'),
        ('Ibuprofen', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'),
        ('Naproxen', 'COc1ccc2cc(C(C)C(=O)O)ccc2c1'),
        ('Diclofenac', 'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl'),
    ],
    'Opioids': [
        ('Morphine', 'OC1=CC=C2CC3N(CCC34CCc5c4cc(O)c(OC)c5)C2=C1'),
        ('Codeine', 'COc1ccc2CC3N(CCC34CCc5c4cc(OC)c(O)c5)C2=c1'),
        ('Tramadol', 'OC1(c2ccccc2)CCCCC1CN(C)C'),
    ],
    'Stimulants': [
        ('Caffeine', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'),
        ('Amphetamine', 'CC(N)Cc1ccccc1'),
        ('Modafinil', 'NC(=O)CS(=O)c1ccc(-c2ccccc2)cc1'),
    ],
    'Antidepressants': [
        ('Fluoxetine', 'CNCCC(c1ccccc1)Oc1ccc(cc1)C(F)(F)F'),
        ('Sertraline', 'CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21'),
        ('Venlafaxine', 'COc1ccc(C2(CCN(C)C)CCCCC2)cc1'),
    ],
    'Antibiotics': [
        ('Ciprofloxacin', 'OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O'),
        ('Amoxicillin', 'CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O'),
    ]
}

# Flatten the data
molecules = []
for category, compounds in molecular_data.items():
    for name, smiles in compounds:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append((name, mol, category))

print(f"Loaded {len(molecules)} molecules from {len(molecular_data)} therapeutic classes")

# Generate multiple fingerprint types
def generate_fingerprints(molecules, fp_types=['morgan', 'maccs']):
    fingerprints = {}

    for fp_type in fp_types:
        print(f"Generating {fp_type} fingerprints...")

        if fp_type == 'morgan':
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for _, mol, _ in molecules]
            fp_array = np.array(fps)

        elif fp_type == 'maccs':
            fps = [MACCSkeys.GenMACCSKeys(mol) for _, mol, _ in molecules]
            fp_array = np.array(fps)

        fingerprints[fp_type] = fp_array

    return fingerprints

fp_types = ['morgan', 'maccs']
fingerprints = generate_fingerprints(molecules, fp_types)

# Extract metadata
names = [name for name, _, _ in molecules]
categories = [cat for _, _, cat in molecules]

print(f"Generated fingerprints:")
for fp_type, fp_array in fingerprints.items():
    print(f"  {fp_type}: {fp_array.shape}")

# Use Morgan fingerprints for main analysis
fps = fingerprints['morgan']

# Dimensionality reduction comparison
methods = ['pca', 'tsne']
embeddings = {}

# PCA
pca = PCA(n_components=2, random_state=42)
pca_embedding = pca.fit_transform(fps)
embeddings['pca'] = pca_embedding

# t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=1000)
tsne_embedding = tsne.fit_transform(fps)
embeddings['tsne'] = tsne_embedding

print("\\nDimensionality reduction completed")

# Create comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Color mapping
unique_cats = list(set(categories))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
cat_colors = dict(zip(unique_cats, colors))

method_names = {'pca': 'PCA', 'tsne': 't-SNE'}

for i, (method, embedding) in enumerate(embeddings.items()):
    ax = [ax1, ax2][i]

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        x, y = embedding[mask, 0], embedding[mask, 1]
        ax.scatter(x, y, c=[cat_colors[cat]], label=cat, alpha=0.8, s=60)

    ax.set_title(f'Chemical Space - {method_names[method]} Projection', fontsize=14)
    ax.set_xlabel(f'{method_names[method]} 1')
    ax.set_ylabel(f'{method_names[method]} 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chemical_space_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Clustering analysis
kmeans = KMeans(n_clusters=len(unique_cats), random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tsne_embedding)

# Evaluate clustering
sil_score = silhouette_score(tsne_embedding, cluster_labels)
print(f"\\nClustering evaluation:")
print(f"Silhouette score: {sil_score:.3f}")
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# Molecular properties analysis
properties = []
for name, mol, category in molecules:
    props = {
        'name': name,
        'category': category,
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
    }
    properties.append(props)

props_df = pd.DataFrame(properties)
props_df['tsne_x'] = tsne_embedding[:, 0]
props_df['tsne_y'] = tsne_embedding[:, 1]

# Property visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

prop_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds']
prop_names = ['Molecular Weight', 'LogP', 'TPSA', 'H-Bond Donors', 'H-Bond Acceptors', 'Rotatable Bonds']

for i, (prop, name) in enumerate(zip(prop_cols, prop_names)):
    if i < len(axes):
        scatter = axes[i].scatter(
            props_df['tsne_x'],
            props_df['tsne_y'],
            c=props_df[prop],
            cmap='viridis',
            alpha=0.8,
            s=60
        )
        axes[i].set_title(f'{name} in Chemical Space', fontsize=12)
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[i], label=name)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('molecular_properties_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Fingerprint comparison
print("\\nFingerprint Type Comparison:")
for fp_type in fp_types:
    X = fingerprints[fp_type]

    # Quick t-SNE for comparison
    tsne_comp = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=500)
    embedding_comp = tsne_comp.fit_transform(X)

    # Evaluate clustering quality
    cat_labels = [unique_cats.index(cat) for cat in categories]
    try:
        sil_score_comp = silhouette_score(embedding_comp, cat_labels)
        print(f"  {fp_type}: Silhouette score = {sil_score_comp:.3f}")
    except:
        print(f"  {fp_type}: Could not compute silhouette score")

print("\\nAnalysis complete!")
print("Generated files:")
print("- chemical_space_analysis.png")
print("- molecular_properties_analysis.png")

# Save results
props_df.to_csv('molecular_analysis_results.csv', index=False)
print("- molecular_analysis_results.csv")