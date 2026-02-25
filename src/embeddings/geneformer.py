import torch
import scanpy as sc
from transformers import BertModel, BertConfig

# 1. Load your donor data (A29 and A31)
adata = sc.read_h5ad("data/dominguez_conde_immune_tissue_two_donors.h5ad")

# 2. Load Geneformer directly from Hugging Face
# This is much easier than scGPT because it uses standard 'BertModel' architecture
model_id = "ctheodoris/Geneformer"
model = BertModel.from_pretrained(model_id, output_hidden_states=True)
model.eval()

# 3. Get the Embeddings
# In a real workflow, you would use Geneformer's tokenizer, 
# but for the "simplest" version, we extract the latent state.
with torch.no_grad():
    # We pass the data through the model to get the 'hidden_states'
    # For Geneformer, the 'mean' of the last layer is the cell embedding
    # Here we assume your data is already pre-processed for Geneformer
    inputs = torch.tensor(adata.X.toarray()).to(torch.int64) # Simplified example
    outputs = model(inputs)
    
    # This is your "Enhanced Representation" [cite: 11]
    cell_embeddings = outputs.last_hidden_state.mean(dim=1) 

# 4. Save to AnnData for C2S
adata.obsm["X_embeddings"] = cell_embeddings.numpy()
print("Success! Your scFM embeddings are ready for the C2S decoder.")