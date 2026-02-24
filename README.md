# Improving-Cell2Sentence-with-Single-Cell-Foundation-Model-Embeddings

Large language model–based approaches such as Cell2Sentence use transformer architec
tures to learn representations from single-cell RNA sequencing (scRNA-seq) data. While
Cell2Sentence was originally introduced for generating natural language descriptions of cells,
its underlying representations can also be applied to downstream tasks such as cell label
prediction. In its original formulation, Cell2Sentence relies on cell representations derived
directly from gene expression profiles, which may not fully capture higher-level biological
structure such as pathways, cell states, or regulatory programs [4]. At the same time, single
cell foundation models (scFMs) such as scGPT, Geneformer, and scFoundation [1, 5, 3]
have been pretrained on large-scale scRNA-seq data and are known to learn biologically
meaningful embeddings of cells. This project investigates whether replacing the original
Cell2Sentence cell representations with embeddings extracted from pretrained scFMs can
improve the quality of generated cellular descriptions. We propose a comparison between
the original Cell2Sentence setup and several variants that use scFM-based embeddings as
input to the text decoder. All models are evaluated on a cell label prediction task using
standard classification metrics.
