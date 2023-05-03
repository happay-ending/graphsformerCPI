# GraphsformerCPI: a graph Transformer for compound-protein interaction  prediction
## Introduction
This is a PyTorch implementation of the research: [a graph Transformer for compound-protein interaction  prediction](https://github.com/happay-ending/graphsformerCPI)

This repository contains a brief description of the paper, source code, data and program run instructions.

----
Predicting compaction-protein interactions (CPIs) plays a crucial role in computer-aided drug design and discovery, and is the most important step to identify drug candidates in virtual screening. With the generation of large amounts of labeled data in the biomedical field and the improvement of computer performance, many in silico methods have been proposed. Unfortunately, the sparse number of proteins with known 3D structures makes these methods overly dependent on manual features, leading to errors. Recent advances in AlphaFold2 have made it possible to accurately compute a large number of protein structures, opening up new opportunities for the deep learning techniques in drug research. Here, we propose an end-to-end deep learning framework termed GraphsformerCPI to significantly improve the efficiency and interpretability of compound-protein interaction prediction. The model treats compounds and proteins as node sequences with spatial structures and learns their feature representations using one-dimensional node sequences and two-dimensional graph structures. The model retrieves the contact maps from the predicted 3D structures of proteins and the molecular graphs of compounds from the structures of small molecules, obtains the node semantic weight matrices of compounds and proteins using a self-attention mechanism, and finally incorporates the graph structural features and semantic features. To learn the relationship features of compounds and proteins, the similarity of atoms and residues is calculated using an inter-attention mechanism. This fusion of structural and semantic information of compounds and proteins has rarely been applied in CPI. Moreover, the attention mechanism is employed intra- and inter-molecules, which makes the proposed model more interpretable than most black-box deep learning methods. We conduct extensive experiments on human, C.elegans, Davis, and KIBA datasets. The effects of the number of layers and dropout on the model performance are explored. The high performance of the model is verified by comparing experiments. The compounds, proteins, and their intrinsic interactions and binding principles in CPI are explained by molecular docking. Our experiments show that GraphsformerCPI can be effectively applied to predict the interactions and binding affinities of compounds and proteins, which has important practical implications for key atom and residue identification, drug candidate discovery, and even downstream pharmaceutical tasks.

----
## Dependencies
```
biopython	1.79
networkx	2.2
numpy	1.21.6
pandas	1.3.5
python	3.7.13
rdkit	2018.09.3
scikit-learn	1.0.2
scipy	1.7.3
seaborn	0.12.2
tensorboard	2.9.0
torch	1.11.0+cu102
torchmetrics	0.11.0
tqdm	4.64.0
```
----
## Using
Run the classified_multi_gpu task program using the following command:
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env classified_multi_gpu.py
```
To specify the number of GPUs to be used you can use the following command. For example, to use the 1st and 4th GPU for training:
```
CUDA_VISIBLE_DEVICES=1,4 python -m torch.distributed.launch --nproc_per_node=2 --use_env classified_multi_gpu.py
```
Run the regression task program using the following command:
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env regression_multi_gpu.py
```
To specify the number of GPUs to be used you can use the following command. For example, to use the 1st and 4th GPU for training:
```
CUDA_VISIBLE_DEVICES=1,4 python -m torch.distributed.launch --nproc_per_node=2 --use_env regression_multi_gpu.py
```

**notes**: You can specify the dataset, epochs, layers, etc. in the run parameters. 

The results are stored in the `log` folder after the program is run, and the training model parameters are stored in `output/model/`. 
