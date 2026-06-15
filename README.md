# PathVI: Pathway-aware Variational Autoencoder for Interpretable Representations of Gene Expression Data Across Multiple Biological Scales

This repository contains the code for the paper *"Pathway-aware Variational Autoencoder for Interpretable Representations of Gene Expression Data Across Multiple Biological Scales"*. 

# Abstract
**Motivation:** Biomarker discovery and molecular subtype classification in cancer are hindered by the high dimensionality and noise of gene expression data, and gene-only deep learning approaches may face limited biological interpretability. Because cancer phenotypes arise from coordinated dysregulation of gene networks, incorporating pathway-level biological knowledge directly into model design offers a principled approach to improve robustness, interpretability, and predictive performance.

**Result:** We present a pathway-aware variational autoencoder (PathVI) that integrates curated gene–pathway structure into latent representation learning. Across multiple transcriptomic datasets, including TCGA-BRCA, GBM, LGG, and cortex data, pathway-level representations significantly improve reconstruction accuracy, reducing bias and error relative to gene-level baselines. In supervised classification tasks, PathVI consistently achieves higher accuracy and greater stability across trials. Low-dimensional embeddings reveal biologically coherent organization of breast cancer subtypes, reflecting known molecular relationships. Importantly, task-dependent gene importance scores provide intrinsic interpretability, with reconstruction emphasizing conserved signaling pathways and classification highlighting subtype-discriminative immune and regulatory genes. These results demonstrate that pathway-aware representation learning unifies predictive performance with biologically grounded interpretability, enabling robust systems-level biomarker discovery.

# Dependencies
- Python 3.12 or higher
- PyTorch 2.0 or higher
- PyTorch Geometric 2.0 or higher
- Pandas
- NumPy
- Seaborn
- Matplotlib

