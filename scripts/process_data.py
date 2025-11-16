"""
Simple data processing script for the ogbn-products`.

It expects CSVs in `data/raw/`:
- node-feat.csv  (no header)
- node-label.csv (no header)
- edge.csv       (no header, two columns: src, dst)

Outputs (saved to both `data/processed/` and
`dataset/ogbn_products/processed/data_processed/`):
- node_feat.npy
- node_label.npy
- edge_index.npy
- adjacency.npz (scipy sparse CSR)

Run: python scripts/process_data.py
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse

# Paths (relative)
RAW_DIR = os.path.join("data", "raw")
OUT_DIRS = [
    os.path.join("data", "processed"),
    os.path.join("dataset", "ogbn_products", "processed", "data_processed"),
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

# Filenames
NODE_FEAT_F = os.path.join(RAW_DIR, "node-feat.csv")
NODE_LABEL_F = os.path.join(RAW_DIR, "node-label.csv")
EDGE_F = os.path.join(RAW_DIR, "edge.csv")

print("Reading input files from:", RAW_DIR)

# Load CSVs (no header assumed)
node_feat = pd.read_csv(NODE_FEAT_F, header=None)
node_label = pd.read_csv(NODE_LABEL_F, header=None)
edge_df = pd.read_csv(EDGE_F, header=None)

# Convert to numpy
node_feat_np = node_feat.values.astype(np.float32)
node_label_np = node_label.values.squeeze().astype(np.int64)
edge_np = edge_df.values.astype(np.int64)

num_nodes = node_feat_np.shape[0]
num_edges = edge_np.shape[0]

print(f"Loaded node features: {node_feat_np.shape}")
print(f"Loaded node labels: {node_label_np.shape}")
print(f"Loaded edges: {edge_np.shape}")

# Basic validations
if node_label_np.shape[0] != num_nodes:
    print("Warning: number of labels does not match number of nodes.")

max_idx = int(edge_np.max())
if max_idx >= num_nodes:
    raise ValueError(f"Edge contains node index {max_idx} >= num_nodes ({num_nodes})")

# Build symmetric/undirected adjacency
src = edge_np[:, 0]
dst = edge_np[:, 1]

both_src = np.concatenate([src, dst])
both_dst = np.concatenate([dst, src])

data = np.ones(both_src.shape[0], dtype=np.uint8)
adj = sparse.csr_matrix((data, (both_src, both_dst)), shape=(num_nodes, num_nodes))
# Remove duplicate entries (sum duplicates)
adj.sum_duplicates()

# Optionally remove self-loops
if adj.diagonal().sum() > 0:
    print("Removing self-loops if present.")
    adj.setdiag(0)
    adj.eliminate_zeros()

# Save outputs to each output directory
for out in OUT_DIRS:
    np.save(os.path.join(out, "node_feat.npy"), node_feat_np)
    np.save(os.path.join(out, "node_label.npy"), node_label_np)
    np.save(os.path.join(out, "edge_index.npy"), edge_np)
    sparse.save_npz(os.path.join(out, "adjacency.npz"), adj)
    print(f"Saved processed files to: {out}")

print("Processing complete.")
print(f"num_nodes={num_nodes}, num_edges(original)={num_edges}, adjacency nnz={adj.nnz}")
