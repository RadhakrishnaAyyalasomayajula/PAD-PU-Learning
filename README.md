# Uncertainty-Aware Graph Representation Learning with 
# Positive-Unlabeled Classification for Biomarker Discovery 
# in Peripheral Artery Disease

## Overview

This repository contains the code and analysis pipeline for the 
manuscript:

**"Uncertainty-aware graph representation learning with 
positive-unlabeled classification for biomarker discovery in 
peripheral artery disease"**

Venkat Ayyalasomayajula, Jelmer M. Wolterink, Kak Khee Yeung

Amsterdam UMC, Amsterdam, The Netherlands  
University of Twente, Enschede, The Netherlands

## Status

This manuscript is currently under review at PLOS Computational 
Biology. Full code and documentation will be uploaded upon 
acceptance of the manuscript.

## Framework Overview

The pipeline comprises four phases:

1. **PPI network construction** — STRING v12.0, ~7,600 proteins, 
   68 known PAD-positive proteins
2. **Unsupervised graph embeddings** — GATv2 encoder with 8 
   self-supervised objectives (DGI, GAE, VGAE, GRACE, SimGRACE, 
   MAE × 3)
3. **Ensemble PU learning** — 5 RN heuristics, 5 classifiers, 
   40 ensemble models, uncertainty quantification
4. **Candidate stratification** — 100 novel PAD biomarker 
   candidates stratified by confidence and novelty, 
   GNNExplainer-based explainability

## Data Availability

The protein-protein interaction network is available from STRING 
v12.0 at https://string-db.org. ProBERT and BioBERT pre-trained 
models are publicly available.

## Contact

For questions regarding the manuscript or code, please contact:  
v.ayyalasomayajula@amsterdamumc.nl

## License

To be added upon acceptance.
