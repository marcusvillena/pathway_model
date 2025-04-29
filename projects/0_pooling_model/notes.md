# ipynb
* 0: testing pooling models
    * 0_0.ipynb
        * main structure, pooling classes, encoder/decoder structure, training module etc.
        * mse/mae ~ std in reconstruction; needs improvement

    * 0_1.ipynb
        * cleaned up code to ./modules
        * INCOMPLETE; troubleshooting 0_0
        * check: simple decoder (MLP only, pathway unaware) to see if it is the bottleneck
        * otherwise, need to revise encoder structure

* 1: testing transformer models
    * 1_0.ipynb
        * manual transformer attention modules (MHA, MQA, GQA)
        * did not realize nn.MultiheadAttention exists :/

    * 1_1.ipynb, 1_2.ipynb
        * 1_1: testing with nn.MultiheadAttention, using path- and gene reconstruction
        * 1_2: simpler decoder, gene reconstruction only
        * INCOMPLETE: need to package these nicely into a module

* 2: benchmarking
    * 2_0.ipynb
        * general benchmarking, GLMs

# model outline
* encoder:
    0. input: graph (x) in (b, n, F)
        * graph data originally in (b * n, F), reshape to (b, n, F)
        * F_0 = gene count
        * F_1 = graph laplacian (positional encoding)

    1. GNN encoding
        * neighbor -> node embedding in (b, n, E)
        * graph attention network with tanh-L1 ('interative') attention
        * no self-loops,prediction from parent nodes (A_in - I)

    2. Set pooling
        * node -> pathway embedding in (b, s, E)
        * pooled-query masked self-attention transformer
        * same space

    3. Global pooling
        * pathway -> sample embedding (z) in (b, E)
        * BERT-like encoder

    4. Reparameterization (optional)
        * variational transformer
        * maps sample embeddings to a distribution

    00. output: sample embedding (z) in (b, E)

* simple decoder:
    0. input: sample embedding (z) in (b, E)

    1. VecExpand
        * MLP outputs (b, n * E), reshape to node embedding -> (b, n, E)
        * optional: MLP inputs, outputs (b, n, E)
        * optional: transformer self-attention
        * optional: FiLM modulation (pre- and/or post-transformer)

    2. ZINB estimation
        * MLP outputs (b, n, k*E), chunk to k ZINB parameters -> k of (b, n, E)
        * ZINB prediction of counts
        * separate MLP for calculating bias

* embedding sizes:
    * encoder
        * node embedding in (b, n, E_node)
        * set embedding in (b, n, E_node)
        * sample embedding in (b, n, E_sample)
            * E_sample may equal E_node if in same space / for simplicity

    * simple decoder
        * sample expansion in (b, n * D_node) -> reshape -> (b, n, D_node)
        * k params in (b, n, k * D_params) -> chunk -> k * (b, n, D_params)
            * k is 3 (mu,theta,pi for ZINB) or 5 (gamma,beta for FiLM)

    * tot. hyperparameters: 18 (model) + 3 (training)
        * 4 = embedding size: E_node, E_sample, D_node, D_params
        * 4 = num_heads per attention model, min. 4: GAT, SetPooling, GlobalPooling, CountsEst
        * 10 = 5x2 = num_layers, layer_size per linear/stack, min. 5: GAT (stack), SetPooling (lin), GlobalPooling (lin), VecExpand (lin), CountsEst (lin)
        * 3 = training: learning rate, batch size, num epochs



# experiments
1. tuning/ablation with simple decoder obj.
    * (do #2 first for reference)

2. benchmarking: counts prediction / reconstruction
    * NB-GLM
        * DESeq2-based GLM

    * ZINB-GLM
        * ZIFA, ZINB-WaVE ?

    * DEAP-MLP
        * absolute sum (add pool) as MLP decoder input

    * MLP, GCN, GAT, ... , others?
        * compare direct vs. pathway (AE)-based

    * perhaps separate table/experiment for ZINB parameters prediction ?

3. downstream: phenotype classification
    * MLP, GCN, GAT, ... , others?
    * can the GLMs classify?

4. applications (time-permitting)
    * embedding visualization (PCA, tSNE, UMAP)
        * plot: patient PCA, coloured by phenotype
        * plot: gene PCA vs. pathway PCA per phenotype

    * differential expression of genes
        * plot: DESeq2 (fold change) vs gene embedding distance/similarity
        * PCA with arrows from one point (phenotype A) to another (phenotype B) of the same gene?
        * if VAE: generative embedding visualization eg visualize synthetic embedding as transition from one phenotype to another

    * differential expression of pathways
        * plot: DEAP (abs. sum) vs. pathway embedding distance/similarity
        * similar to DEA genes

    * set enrichment vs. set attention
        * GSEA (enrichment) vs. set enrichment vs. set (global pooling) attention
        * set attention in reconstruction vs in classification

    * case study: GIAtConv attention vs. literature (KEGG) pathway knoweldge
        * compare attention to known functions
        * plot: attention values distribution per edge type

5. etc. (optional)
    * simple decoder vs. pathway decoder multi-objective
        * compare pathway embedding consistency & prediction quality

    * downstream: protein expression prediction

    * applications: non-annotated genes
        * mlp embedding of genes not annotated in pathway
        * if in same space (?) cosine similarity and distance to known pathways

    * applications: set learning task
        * user-defined k number of sets
        * make node-set mask M (nodes, k) a learnable parameter in training; softmax in set dimension
        * must modify loss function ???
        * analysis/validation: cosine similarity and overlap to known pathways
        * if dynamic (custom M per sample): compare common sets per phenotype

# to-do
1. benchmarking tool
    * experiment (w/ autoplot) class
    * implement GLMs and MLP for comparison

2. model
    * Encoder: 
        * non-lazy AttentionPooling class
            * masked query only; use original kv (no qkv lin)
            * togglable masked attention

    * SimpleDecoder:
        * Vec2Node module
            * (b,E') -> MLP -> (b,n*E) -> reshape -> (b,n,E)
            * optional: MLP(b,n,E), FiLM (where?)

        * NodeDecoder module
            * (b,n,E) -> MLP or transformer -> (b,n,3*E) -> chunk -> mu,theta,pi in (b,n,E)
            * 3*(b,n,E) -> 3*MLP per mu,theta,pi -> mu,theta,pi in (b,n)

        * GenePredictor module
            * use mu,theta,pi to predict expected gene count
            * input z to (MLP or transformer) to generate FiLM bias
            * modulate expected gene count to get predicted gene count
            * can be merged with NodeDecoder; e.g. chunk (b,n,5*E) to mu,theta,pi (ZINB) and beta,gamma (FiLM)
            * output both expected (smooth) and predicted (noisy)
            * 


# stats
* Loss curves
    (train, val, test) per epoch
* Calibration curve
    x vs. x_recon
* Dropout curve
    predicted prop. 0 vs. true prop. 0
* 
* GLM parameter estimations
    * one-way anova for each parameter (mu, theta, pi)
    * visualize: boxplot, distribution
    * predicted (train) vs. actual (test) ???

# etc
bahdanau attention
luong attention
SiGAT

CCA
Barlow tiwns
VICReg

edge feature = parent node feature
* attention visualization
    * per-pathway or per-node
        * pathway map: pathway_idx (0 to 305) -> pathway_name
        * node name map: node_idx (0 to 4383) -> gene_name (not ensembl)


* DESeq2: unscaled data with NB trained on NB NLL
* limma: log2 data with linear trained on MSE (OLS)