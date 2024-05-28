# Summary
This repository is easy code for [InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning, ICML 2024](https://arxiv.org/abs/2303.04947)\
It is based on the Docker image `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel`.\
After installing the Docker image, you can reproduce the experiment by running the container using `run_container.sh`.

# Preparing

### Running the Container

```sh
sh run_container.sh
```
If the volume mount path is incorrect, please modify the -v parameter in the Docker command within the run_container.sh file.

###  Installing MLflow
After running the container, install MLflow inside the container
```sh
pip install mlflow
```

# File structure
All source code is located in the src folder.\
The main execution files for performing experiments are named in the format res18_cifar_xx_xx.py.\
The code for Pruning Policy and Dataset is located in the src/utils directory.\
The hyper-parameters used in training are in the config directory.\
Pruning probability and Annealing values can be adjusted through the PruningPolicy parameters in src/utils/policy.py.


**res18_cifarXX_whole.py**: Experiment code for training on the entire dataset.\
**res18_cifarXX_ib.py**: Experiment code for training using InfoBatch.\
**res18_cifarXX_ib_ma.py**: Experiment code for performing InfoBatch using the Moving Average threshold.\
**res18_cifarXX_ib_rev.py** Experiment code for training pruned samples using InfoBatch.

# Running Experiments
Running each experiment code (res18_cifarXX_XX_XX.py) will log the experiment results in MLflow.
You can view the experiment logs using the MLflow UI.
```sh
mlflow ui
```
