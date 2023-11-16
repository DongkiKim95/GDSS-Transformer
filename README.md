In this repository, we implement the *Graph Diffusion via the System of SDEs* (GDSS) **using Graph Transformer**.

Paper: [Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations](https://arxiv.org/abs/2202.02514) (ICML 2022).

Original Code Repository: https://github.com/harryjo97/GDSS

## Dependencies

Please create an environment with **Python 3.9.15** and **Pytorch 1.12.1**, and run the following command to install the requirements:
```
pip install -r requirements.txt
conda install pyg -c pyg
conda install -c conda-forge graph-tool=2.45
conda install -c conda-forge rdkit=2022.03.2
```


## Running Experiments

### 1. Preparations

We provide four **general graph datasets** (Planar and SBM) and two **molecular graph datasets** (QM9 and ZINC250k). 

Download the datasets from the following links and <u>move the dataset to `data` directory</u>:

+ Planar (`planar_64_200.pt`): [https://drive.google.com/drive/folders/13esonTpioCzUAYBmPyeLSjXlDoemXXQB?usp=sharing](https://drive.google.com/drive/folders/13esonTpioCzUAYBmPyeLSjXlDoemXXQB?usp=sharing)

+ SBM (`sbm_200.pt`): [https://drive.google.com/drive/folders/1imzwi4a0cpVvE_Vyiwl7JCtkr13hv9Da?usp=sharing](https://drive.google.com/drive/folders/1imzwi4a0cpVvE_Vyiwl7JCtkr13hv9Da?usp=sharing)

We provide the commands for generating general graph datasets as follows:

```
python data/data_generators.py --dataset <dataset> --mmd
```
where `<dataset>` is one of the general graph datasets: `planar` and `sbm`.
This will create the `<dataset>.pkl` file in the `data` directory.

To preprocess the molecular graph datasets for training models, run the following command:

```sh
python data/preprocess.py --dataset ${dataset_name}
python data/preprocess_for_nspdk.py --dataset ${dataset_name}
```

For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

```sh
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

### 2. Training

We provide the commands for the following tasks: Generic Graph Generation and Molecule Generation.

To train the score models, first modify `config/${dataset}.yaml` accordingly, then run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type train --config ${train_config} --seed ${seed}
```

for example, 

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --type train --config planar --seed 42
```
and
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type train --config qm9 --seed 42
```

### 3. Generation and Evaluation

To generate graphs using the trained score models, run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type sample --config planar
```
or
```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type sample --config sample_qm9
```


## Pretrained checkpoints

We provide checkpoints of the pretrained models in the follwoing links:
+ Planar: https://drive.google.com/drive/folders/18P_W6B-aBul_OFkIBsl9aPdT906CPDfZ?usp=drive_link
+ SBM: https://drive.google.com/drive/folders/1LIUNf96IYefMfkospvbqmcvvPYBSukgP?usp=drive_link
+ QM9: https://drive.google.com/drive/folders/1loFz_DIzt6JGAvUoB2zvTV9TvuX34A3G?usp=drive_link
+ ZINC250k: https://drive.google.com/drive/folders/19WBDXDLph_QdA7T6MfEWkpmGPujpgPZ4?usp=drive_link

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@article{jo2022GDSS,
  author    = {Jaehyeong Jo and
               Seul Lee and
               Sung Ju Hwang},
  title     = {Score-based Generative Modeling of Graphs via the System of Stochastic
               Differential Equations},
  journal   = {arXiv:2202.02514},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.02514}
}
```

