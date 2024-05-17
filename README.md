# Hi-GMAE: Hierarchical Graph Masked Autoencoders

## Overview

![image](./imgs/image.png)

This paper presents **Hi-GMAE**, a novel multi-scale GMAE framework designed to handle the hierarchical structures within graphs. 
Diverging from the standard graph neural network (GNN)
used in GMAE models, Hi-GMAE modifies its encoder and decoder
into hierarchical structures. This entails using GNN at the finer scales
for detailed local graph analysis and employing a graph transformer
at coarser scales to capture global information. 
### Python environment setup with Conda

```
conda create -n himae python=3.11
conda activate himae
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install ogb
pip install pygsp
pip install scipy
pip install tensorboardX
pip install matplotlib
pip install sortedcontainers
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Running Hi-GMAE

Running unsupervised graph classification:

```
conda activate himae
# Running Hi-GMAE tuned hyperparameters for PROTEINS.
sh ./scripts/protein.sh 
# Running Hi-GMAE tuned hyperparameters for COLLAB.
sh ./scripts/collab.sh 
# Running Hi-GMAE tuned hyperparameters for D&D.
sh ./scripts/dd.sh 
```
Running transfer learning on molecular property prediction:

```
conda activate himae
cd transfer_learning
# Pretraining Hi-GMAE on ZINC15.
python pretraining.py
# Finetuning Hi-GMAE on MoleculeNet datasets.
i.e. finetune on BACE
sh ./scripts/bace.sh
```

Supported datasets:

- TUDataset: `NCI1`, `PROTEINS`, `D&D`, `IMDB-BINARY`, `IMDB-MULTI`, `COLLAB`, `REDDIT-BINARY`
- MoleculeNet: `BBBP`, `Tox21`, `ToxCast`, `SIDER`, `ClinTox`, `MUV`, `HIV`, `BACE` 

### Baselines

- Infomax:https://github.com/snap-stanford/pretrain-gnns
- ContextPred:https://github.com/snap-stanford/pretrain-gnns                                            
- AttrMasking:https://github.com/snap-stanford/pretrain-gnns
- GCC:https://github.com/THUDM/GCC
- GraphCL:https://github.com/Shen-Lab/GraphCL
- SimGrace:https://github.com/junxia97/SimGRACE
- JOAO:https://github.com/Shen-Lab/GraphCL_Automated  
- GraphLoG:https://github.com/DeepGraphLearning/GraphLoG
- RGCL:https://github.com/lsh0520/rgcl
- S2GAE:https://github.com/qiaoyu-tan/S2GAE
- GraphMAE:https://github.com/THUDM/GraphMAE
- GraphMAE2:https://github.com/thudm/graphmae2
- Mole-BERT:https://github.com/junxia97/mole-bert
## Datasets

Unsupervised graph classification datasets mentioned above will be downloaded automatically using PyG's API when running the code. 

Dataset for molecular property prediction can be found [here](https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). After downloading, unzip it and put it in `transfer_learning/datasets`

Hi-GMAE is built using [PyG](https://www.pyg.org/) and [GraphMAE](https://github.com/THUDM/GraphMAE/tree/main). 
