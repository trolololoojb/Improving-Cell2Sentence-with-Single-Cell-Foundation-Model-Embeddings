import torch
import scanpy as sc
import numpy as np
from transformers import BertModel
from tqdm import tqdm

# 1. Load your donor data
adata = sc.read_h5ad("data/dominguez_conde_immune_tissue_two_donors.h5ad")

# 2. Pre-process: Select the top genes per cell to fit the 4096 limit
# We will keep the 4096 highest-expressed genes per cell to ensure we don't exceed the model's max_position_embeddings
MAX_LEN = 4096 
if adata.n_vars > MAX_LEN:
    print(f"Subsetting genes from {adata.n_vars} to {MAX_LEN} to fit model limits...")
    # This keeps only the genes that are expressed in this dataset
    sc.pp.filter_genes(adata, min_cells=3)
    # If still over 4096, take the most variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=MAX_LEN, flavor='seurat_v3', subset=True)

# 3. Load Geneformer
model_id = "ctheodoris/Geneformer"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertModel.from_pretrained(model_id, output_hidden_states=True).to(device)
model.eval()

# 4. Memory-Safe Minibatch Loop
batch_size = 12 
n_cells = adata.shape[0]
all_embeddings = np.zeros((n_cells, 256)) # Geneformer-10M is 256-D

print(f"Extracting representations for {n_cells} cells...")
with torch.no_grad():
    for i in tqdm(range(0, n_cells, batch_size)):
        idx = range(i, min(i + batch_size, n_cells))
        
        # Convert ONLY the current slice to dense and move to GPU
        # Using .toarray() on the small slice avoids the crash
        batch_data = torch.tensor(adata.X[idx].toarray()).to(torch.int64).to(device)
        
        # Truncate sequence length dimension if it still exceeds 4096
        if batch_data.shape[1] > MAX_LEN:
            batch_data = batch_data[:, :MAX_LEN]
            
        # Forward pass
        outputs = model(batch_data)
        
        # MEAN POOLING: Get one 256-D vector per cell
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings[idx] = embeddings

# 5. Save results
adata.obsm["X_embeddings"] = all_embeddings
print("Success! 'Enhanced representations' extracted without exceeding length limits.")