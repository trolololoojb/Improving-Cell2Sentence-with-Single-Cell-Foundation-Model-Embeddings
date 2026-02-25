import torch
import scanpy as ad
from scgpt.models import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab

# 1. Load your subsetted dataset (Donors A29 and A31)
# Assuming your adata is already filtered and normalized as per Domínguez Conde et al.
adata = ad.read_h5ad("../../dominguez_conde_immune_tissue_two_donors.h5ad")

# 2. Load scGPT Pretrained Model and Vocabulary
model_dir = "../../models/scGPT"
vocab = GeneVocab.from_file(f"{model_dir}/vocab.json")
model = TransformerModel.from_pretrained(model_dir)
model.to("cuda")
model.eval()  # Set to evaluation mode

# 3. Align Gene Names
# Map the gene names in your dataset to the scGPT vocabulary
map_raw_id_to_vocab(adata, vocab)

# 4. Extract Embeddings
# We use the model to encode the cells into a fixed-size latent space
with torch.no_state_grad():
    # This function typically handles tokenization and forward pass
    # It returns a cell-by-embedding matrix (e.g., 29773 x 512)
    embeddings = model.encode(
        adata, 
        batch_size=64, 
        device="cuda",
        return_numpy=True
    )

# 5. Store the "richer" representations back in AnnData [cite: 17]
adata.obsm["X_scGPT"] = embeddings
print(f"Embeddings extracted. Shape: {adata.obsm['X_scGPT'].shape}")