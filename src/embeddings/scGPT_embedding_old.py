import torch
import scanpy as ad
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import preprocess_utils

# 1. Load your dataset (the .h5ad file from your Google Drive link)
adata = ad.read_h5ad("data/dominguez_conde_immune_tissue_two_donors.h5ad")

# 2. Load scGPT Vocabulary and Model
# Ensure you have downloaded the scGPT checkpoint (e.g., scGPT_human_heart or whole_human)
model_dir = "models/scGPT"
vocab = GeneVocab.from_file(model_dir + "vocab.json")
model = TransformerModel.from_pretrained(model_dir)
model.eval() # Set to evaluation mode
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 3. Preprocess: Align genes and bin expression values
# scGPT requires discrete bins rather than raw normalized counts [cite: 16, 22]
adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var_names]
adata = adata[:, adata.var["id_in_vocab"] >= 0] # Filter genes not in vocab

# Binning continuous expression into discrete levels (default is 51 bins)
preprocess_utils.binning(adata, n_bins=51)

# 4. Extraction Loop (The "Frozen" Encoder)
all_embeddings = []

with torch.no_grad(): # Ensure the encoder remains frozen [cite: 31, 33]
    for cell_batch in adata.chunk_X(batch_size=64):
        # Convert binned data to tensors
        src_key_padding_mask = (cell_batch == 0)
        
        # Forward pass to get hidden states
        output = model(
            src_key_padding_mask=src_key_padding_mask,
            batch_data=cell_batch,
            return_embeddings=True
        )
        
        # Extract the cell-level embedding (often the first token or mean pooling)
        # Your goal is a fixed-size cell embedding for the decoder 
        cell_emb = output["cell_embedding"] 
        all_embeddings.append(cell_emb.cpu())

# 5. Save for Cell2Sentence Decoder
import numpy as np
adata.obsm["X_scGPT"] = np.concatenate(all_embeddings, axis=0)
adata.write("immune_subset_with_embeddings.h5ad")