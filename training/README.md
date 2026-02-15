# PotSim: A Large-Scale Simulated Dataset for Benchmarking AI Techniques on Potato Crop

This repository is the suplemental code for PotSim Dataset [here](https://doi.org/10.7910/DVN/GQMDOV) and paper.

## Model Training Setup

The `train_model` function in `train.py` serves as the core function used in our training pipeline for our experiments. This function handles:

- **Training and Validation**: Runs training and validation on respective splits for each epoch
- **Mixed Precision Training**: Uses PyTorch's AMP for faster training
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Automatic Checkpointing**: Saves best model based on validation performance
- **Comprehensive Logging**: Records training/validation metrics for analysis
- **Learning Rate Scheduling**: Reduces learning rate when performance plateaus

### Parameters

| Argument              | Type                                    | Optional | Description                                                                                                                            |
| --------------------- | --------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`          | `str`                                   | No       | Name identifier for the model (used for saving files).                                                                                 |
| `model`               | `torch.nn.Module`                       | No       | PyTorch model instance to be trained.                                                                                                  |
| `train_loader`        | `torch.utils.data.DataLoader`           | No       | DataLoader for the training dataset.                                                                                                   |
| `val_loader`          | `torch.utils.data.DataLoader`           | No       | DataLoader for the validation dataset.                                                                                                 |
| `tgt`                 | `str`                                   | No       | Target variable name (used for organizing saved models and logs).                                                                      |
| `lr`                  | `float`                                 | No       | Initial learning rate for the optimizer.                                                                                               |
| `max_epochs`          | `int`                                   | No       | Maximum number of epochs to train the model.                                                                                           |
| `min_tgt`             | `float` or `int`                        | No       | Minimum value of the `tgt` (e.g., for denormalization or R² calculation).                                                              |
| `max_tgt`             | `float` or `int`                        | No       | Maximum value of the `tgt` (e.g., for denormalization or R² calculation).                                                              |
| `criterion`           | `torch.nn.Module`                       | Yes      | Loss function to be used for training. Defaults to `torch.nn.MSELoss()`                                                                |
| `optimizer`           | `torch.optim.Optimizer`                 | Yes      | Optimizer to be used for training Defaults to `torch.optim.SGD(params, lr, momentum=0.9)`                                              |
| `scheduler`           | `torch.optim.lr_scheduler._LRScheduler` | Yes      | Learning rate scheduler. Defaults to `ReduceLROnPlateau(optimizer, factor=0.5, patience=5)`                                            |
| `early_stop_patience` | `int`                                   | Yes      | Epochs with no improvement on validation loss before stopping training. Defaults to Defaults to `10`                                   |
| `device`              | `torch.device`                          | Yes      | Torch device (`cpu` or `cuda`) on which to train the model. Defaults to `torch.device("cuda" if torch.cuda.is_available() else "cpu")` |
| `min_loss_reduction`  | `float`                                 | Yes      | Minimum reduction in validation loss to be considered an improvement. Defaults to `Defaults to 1e-4`                                   |

### Usage Example

```python
train_model(
    model_name="LSTM",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    tgt=target_variable,
    lr=0.001,
    max_epochs=100,
    min_tgt=0,
    max_tgt=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

## Output

The training function generates:

- Model checkpoint files (`.pth`) saved at the best validation performance
- CSV logs containing per-epoch metrics (loss, R², learning rate)

## Requirements

Refer to the main repository page: [here](https://github.com/GatorSense/PotSim)
