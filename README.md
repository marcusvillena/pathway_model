# PathVI: Pathway-aware Variational Autoencoder for Interpretable Representations of Gene Expression Data Across Multiple Biological Scales

This repository contains the code for the paper *"Pathway-aware Variational Autoencoder for Interpretable Representations of Gene Expression Data Across Multiple Biological Scales"*. 

## Abstract
**Motivation:** Biomarker discovery and molecular subtype classification in cancer are hindered by the high dimensionality and noise of gene expression data, and gene-only deep learning approaches may face limited biological interpretability. Because cancer phenotypes arise from coordinated dysregulation of gene networks, incorporating pathway-level biological knowledge directly into model design offers a principled approach to improve robustness, interpretability, and predictive performance.

**Result:** We present a pathway-aware variational autoencoder (PathVI) that integrates curated gene–pathway structure into latent representation learning. Across multiple transcriptomic datasets, including TCGA-BRCA, GBM, LGG, and cortex data, pathway-level representations significantly improve reconstruction accuracy, reducing bias and error relative to gene-level baselines. In supervised classification tasks, PathVI consistently achieves higher accuracy and greater stability across trials. Low-dimensional embeddings reveal biologically coherent organization of breast cancer subtypes, reflecting known molecular relationships. Importantly, task-dependent gene importance scores provide intrinsic interpretability, with reconstruction emphasizing conserved signaling pathways and classification highlighting subtype-discriminative immune and regulatory genes. These results demonstrate that pathway-aware representation learning unifies predictive performance with biologically grounded interpretability, enabling robust systems-level biomarker discovery.

## Dependencies
- Python 3.12 or higher
- PyTorch 2.0 or higher
- PyTorch Geometric 2.0 or higher
- Pandas
- NumPy
- Seaborn
- Matplotlib

## Usage
### Data Importing and Preprocessing
```python
import modules.data as d
from pathlib import Path

dataset_dir = Path('/datasets/')

brca = d.TCGA(
    tcga_project = 'BRCA',
    tcga_dir = dataset_dir/'tcga',
    subtype_col = 'paper_BRCA_Subtype_PAM50',
    drop = ['Normal', 'Primary Tumor', 'Metastatic'],
    gene_name_path = dataset_dir/'other'/'name2ensg.csv',
    keep_noname = False,
)

kegg = d.KEGG(
    relation_filepath=dataset_dir/'other'/'relation_ohe.csv', 
    counts_data=brca,
)

dataset = d.GraphDataset(brca, kegg, kegg)
```

### Training a Model
```python
# Training a model
from modules.layers import MultiheadSetPooling
from modules.loss import MultiLoss, KLDLoss
from modules.model import BaseAutoencoder
from modules.norm import LogCounts
from modules.train import Loader
from modules.trainers import ReconstrTrainer
from modules.utils import dict_summary

import torch
import torch.nn as nn

loader = Loader(
    dataset,
    device=device,
    batch_size=128,
)

trainer = ReconstrTrainer(
    lr=1e-3, 
    norm_class=LogCounts,
    out_keys={'input':'x_preds', 'target':'x_t', 'mu':'z_mu', 'logvar':'z_logvar'},
    loss_class=MultiLoss,
    loss_kwargs={
        'loss_classes': [nn.MSELoss, KLDLoss],
        # 'ema_norm': True,
        'loss_weights': (1,1e-5),
        # 'warmup':40*7
    }
)

ae = BaseAutoencoder(
    dataset=dataset,
    embed_dim=128,
    num_heads=4,
    method='set',

    norm_class=LogCounts,
    encoder_class=nn.Linear,
    pooling_class=MultiheadSetPooling,
    variational=True,

    hidden_dims=2,
    act_fn=nn.ReLU,
    norm_fn='layer',

    norm_kwargs={'libnorm':True, 'znorm':True, 'learnable':True}
)

trainer.run(
    model=ae,
    loader=loader,
    num_epochs=50,
    report_metrics=['loss','kld','rmse','mae','r2'],
    verbose=True
)

# output preview
out = trainer.model(_batch, need_weights=True)
# out = ae(_batch, need_weights=False)
if isinstance(out, torch.Tensor):
    print(out.shape)
elif isinstance(out, dict):
    print(dict_summary(out))
else:
    print(out)

```