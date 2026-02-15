# PotSim: A Large-Scale Simulated Dataset for Benchmarking AI Techniques on Potato Crop

This repository is the official suplemental code for [PotSim Dataset](https://doi.org/10.7910/DVN/GQMDOV) and paper []()

## Model Architectures

### LinearRegression
A simple Linear Regression model represnting a linear function `y = wx + b`, where `w` is weight vector and `b` is  a bias vector. It uses PyTorch's `nn`.Linear module to implement transformation.

**Example Usage:**
```python
model = LinearRegression(input_dim=10)
```

**Arguments:**
| Argument    | Type  | Optional | Default | Description                      |
| ----------- | ----- | -------- | ------- | -------------------------------- |
| `input_dim` | `int` | No       |         | Dimentionality of input features |

---

### MLP
A Multi-Layer Perceptron (MLP) model i.e. feedforward neural network using PyTorch's `nn.Module` with adjustable layers, hidden sizes, and dropout.


**Example Usage:**
```python
model = MLP(input_dim=10, hidden_size=64, num_layers=2, dropout=0.2)
```

**Arguments:**
| Argument      | Type    | Optional | Default | Description                           |
| ------------- | ------- | -------- | ------- | ------------------------------------- |
| `input_dim`   | `int`   | No       |         | Number of input features              |
| `hidden_size` | `int`   | Yes      | `64`    | Size of hidden layers. Defaults to 64 |
| `num_layers`  | `int`   | Yes      | `2`     | Number of hidden layers               |
| `dropout`     | `float` | Yes      | `0.2`   | Dropout rate                          |

---

### CNN
A 1D Convolutional Neural Network model that consists three convolutional blocks followed by adaptive average pooling and a final linear layer for prediction. Each convulational block includes `Conv1D`, `ReLU` Activation, `BatchNorm`, and `Dropout` (except the last block).

**Example Usage:**
```python
model = CNN1D(input_dim=10, hidden_size=64, kernel_size=3, padding=1, dropout=0.2)
```

**Arguments:**
| Argument      | Type    | Optional | Default | Description                               |
| ------------- | ------- | -------- | ------- | ----------------------------------------- |
| `input_dim`   | `int`   | No       |         | Number of input features (channels)       |
| `hidden_size` | `int`   | Yes      | `64`    | Number of filters in the first conv layer |
| `kernel_size` | `int`   | Yes      | `3`     | Size of the convolving kernel             |
| `padding`     | `int`   | Yes      | `1`     | Padding added to both sides of the input  |
| `dropout`     | `float` | Yes      | `0.2`   | Dropout rate                              |

---

### TCN

A Temporal Convolutional Network (TCN) model composed of a sequence of dilated causal convolutional blocks (`TCNBlock`s), each with increasing dilation factors (`2^i`) to capture long-range temporal dependencies. Each `TCNBlock` consists of convolution, `BatchNorm1d` normalization, `ReLU` activation, and dropout layers. The outputs are globally averaged and passed through a linear layer for final prediction.

**Example Usage:**
```python
model = TCN(input_dim=10, num_channels=[32, 16, 8], kernel_size=3, dropout=0.2)
```

**Arguments:**
| Argument       | Type        | Optional | Default       | Description                                       |
| -------------- | ----------- | -------- | ------------- | ------------------------------------------------- |
| `input_dim`    | `int`       | No       |               | Number of input features (channels)               |
| `num_channels` | `List[int]` | Yes      | `[32, 16, 8]` | Number of feature channels in each residual block |
| `kernel_size`  | `int`       | Yes      | `3`           | Size of the convolving kernel                     |
| `dropout`      | `float`     | Yes      | `0.2`         | Dropout rate                                      |

---

### LSTM

A model with stacked stacked `LSTM` layers for learning temporal dependencies in sequential data. The model contains multiple `LSTM` layers with `dropout`, followed by layer normalization and a fully connected `two-layer` head with `ReLU` activation and `dropout`.

**Example Usage:**
```python
model = LSTM(input_dim=10, hidden_size=64, num_layers=2, dropout=0.2)
```

**Arguments:**
| Argument      | Type    | Optional | Default | Description                                 |
| ------------- | ------- | -------- | ------- | ------------------------------------------- |
| `input_dim`   | `int`   | No       |         | Number of input features                    |
| `hidden_size` | `int`   | Yes      | `64`    | Number of features in the hidden layer      |
| `num_layers`  | `int`   | Yes      | `2`     | Number of recurrent layers (stacked if > 1) |
| `dropout`     | `float` | Yes      | `0.2`   | Dropout rate                                |

---

### EncoderOnlyTransformer

A transformer-based model utilizing only the encoder stack for sequence modeling. The input features are projected to `d_model` dimensions and then combined with positional encoding. The encoder contains multiple encoder layers with multi-head self-attention and feed-forward sublayers (of size `d_model * 4`) and a `dropout`. The model predicts the target using the output of the last token passed through a final linear layer.

**Example Usage:**
```python
model = EncoderOnlyTransformer(input_dim=10, nhead=4, num_layers=2, d_model=128, dropout=0.2)
```

**Arguments:**
| Argument      | Type  | Optional | Default | Description                                   |
| ------------- | ----- | -------- | ------- | --------------------------------------------- |
| `input_dim`   | `int` | No       |         | Number of input features                      |
| `nhead`       | `int` | Yes      | `4`     | Number of heads in MultiHeadAttention         |
| `num_layers`  | `int` | Yes      | `2`     | Number of sub-encoder-layers in the encoder   |
| `d_model`     | `int` | Yes      | `128`   | number of expected features as encoder inputs |
| `dropout`     | `float` | Yes      | `0.2`   | Dropout rate                                  |
| `max_seq_len` | `int` | Yes      | `5000`  | Maximum length for Positional Engcoding       |
