# PotSim: A Large-Scale Simulated Dataset for Benchmarking AI Techniques on Potato Crop

This repository contains the official implementation associated with this paper. 
As this is a double-blind submision- we provide a subset of our large scale data in `data\data_subset` 

## Description:

PotSim is a large-scale simulated agricultural dataset specifically designed for AI-driven research on potato cultivation. This dataset is grounded in real-world crop management scenarios and extrapolated to approximately 4.9 million hypothetical crop management scenarios. It encompasses diverse factors including varying planting dates, fertilizer application rates and timings, irrigation strategies, and 24 years of weather data. The resulting dataset comprises over 675 million daily simulation records, offering an extensive and realistic framework for agricultural AI research.

---

## Repository Items and Usage:

The repository contains four  main files `example.ipynb`, `plots.ipynb`, `run.py` and `Forecasting_inference.ipynb` . To reproduce the results corresponding to regression type predictions-that are train/test results presented in the paper, we provide `run.py`, which can be executed over a command line interface or terminal. To follow a step by step procedure and work with our dataset, we provide `example.ipynb`, a jupyter notebook template, which act as a starting point for further exploration. To make it easier to visualize and plot the results, we have provided `plots.ipynb`, a jupyter notebook template, which contains few example plots and can be edited according the requirements.To run the forecasting models and to see the inference on foundational time series models like Chronos and MOIRAI, then please go to `Forecasting_inference.ipynb` 

---



## Directory Structure:

| Directory Name       | Description                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| `data`               | Contains all datasets required for experiments and analyses.                                 |
| `data/potsim_yearly` | Default location for yearly dataset files utilized in the study.                             |
| `models`             | Houses all model architecture definitions and related scripts.                               |
| `outputs`            | Default directory for saving model checkpoints, logs, and results generated during training. |
| `saves`              | Includes saved model states and checkpoints for all trained models and variables, as well as a Python script from an initial forecasting experiment.|
| `testing`            | Includes scripts and functions for evaluating model performance and generating metrics.      |
| `training`           | Contains training routines, configuration files, and code for model optimization.            |
| `utils`              | Utility functions for data preprocessing, splitting, and model configuration management.     |
| `utils/potsimloader` | Specialized utilities for efficient data loading and processing workflows.                   |

<!-- Stores pre-trained model states and checkpoints from trained models and experiments referenced in the paper.-->

----

## Requirements:

To install the requirements:

```bash
conda env create -f environment.yml
conda activate potsim_env
```

Depending on the version of `CUDA` on your system, install `PyTorch v2.5.1` from official PyTorch source at [https://pytorch.org](https://pytorch.org/get-started/previous-versions/)

```bash
# Example for cuda-version 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

To allow on gpu metrics and display the model parameters clearly

```bash
pip install torchmetrics==1.7.1 torchinfo==1.8
```

If your system is not set up with `conda` package manager, then please visit [https://www.anaconda.com/download](https://www.anaconda.com/download/success#miniconda) to install `Miniconda` accoding to your operating system and then continue by installing the requirements from above.




----

## Usage: `run.py`

The script supports two main commands: `train` and `test`.

- Make sure your datasets are in the `.parquet` format and accessible by the script at `data` folder.
- For more details on available target variables and models, check the code or add a `--help` flag:

```bash
python run.py --help
python run.py train --help
python run.py test --help
```

### 1. Train a Model

```bash
python run.py train -tgt -m  [options]
```

**Arguments:**

| Argument                    | Type  | Optional | Default       | Description                                      |
| --------------------------- | ----- | -------- | ------------- | ------------------------------------------------ |
| `-tgt`, `--target`          | str   | No       |               | Target variable to predict. Choices: _see below_ |
| `-m`, `--model`             | str   | No       |               | Model type to use. Choices: _see below_          |
| `-tdata`, `--train_dataset` | str   | Yes      | `train_split` | Training dataset split                           |
| `-vdata`, `--val_dataset`   | str   | Yes      | `val_split`   | Validation dataset split                         |
| `-bs`, `--batch_size`       | int   | Yes      | `256`         | Batch size                                       |
| `-lr`, `--learning_rate`    | float | Yes      | `0.005`       | Learning rate                                    |
| `-ep`, `--epochs`           | int   | Yes      | `100`         | Maximum number of epochs                         |
| `-sl`, `--seq_len`          | int   | Yes      | `15`          | Sequence length (for sequence models)            |
| `-d`, `--device`            | str   | Yes      | `None`        | Device: `cpu` or `cuda`                          |

**Example:**

```bash
python run.py train -tgt="NTotL1" -m="lstm" -tdata="train_split" -vdata="val_split" -bs=256 -lr=0.001 -ep=10 -sl=15 -d="cuda"
```

---

### 2. Test a Model

```bash
python run.py test -tgt  -m  -data  [options]
```

**Arguments:**

| Argument               | Type | Optional | Default | Description                                                     |
| ---------------------- | ---- | -------- | ------- | --------------------------------------------------------------- |
| `-tgt`, `--target`     | str  | No       |         | Target variable to predict. Choices: _see below_                |
| `-m`, `--model`        | str  | No       |         | Model type to use. Choices: _see below_                         |
| `-data`, `--dataset`   | str  | No       |         | Dataset to run test on                                          |
| `-mdir`, `--model_dir` | str  | Yes      | `saves` | Directory where trained models are saved (`outputs` or `saves`) |
| `-bs`, `--batch_size`  | int  | Yes      | `256`   | Batch size                                                      |
| `-sl`, `--seq_len`     | int  | Yes      | `15`    | Sequence length (for sequence models)                           |
| `-d`, `--device`       | str  | Yes      | `None`  | Device: `cpu` or `cuda`                                         |

**Example:**

```bash
python run.py test -tgt="NTotL1" -m="lstm" -data="test_split" -mdir="saves" -bs=256 -sl=15 -d="cuda"
```

---

## Results:

`R2` Metrics for Different Models

| Target     | CNN1D | Transformer | LSTM  | LinearRegression | MLP   | TCN   |
| ---------- | ----- | ----------- | ----- | ---------------- | ----- | ----- |
| `NLeach`   | 0.432 | -0.02       | 0.343 | 0.002            | 0.014 | 0.265 |
| `NPlantUp` | 0.803 | 0.733       | 0.794 | 0.322            | 0.753 | 0.791 |
| `NTotL1`   | 0.843 | 0.764       | 0.831 | 0.481            | 0.779 | 0.823 |
| `NTotL2`   | 0.861 | 0.799       | 0.849 | 0.489            | 0.792 | 0.843 |
| `SWatL1`   | 0.973 | 0.949       | 0.972 | 0.620            | 0.841 | 0.950 |
| `SWatL2`   | 0.944 | 0.783       | 0.928 | 0.700            | 0.816 | 0.914 |
