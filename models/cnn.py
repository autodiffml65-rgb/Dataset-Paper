import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """A 1D Convolutional Neural Network model that consists three convolutional blocks
    followed by adaptive average pooling and a final linear layer for prediction. Each
    convulational block includes Conv1D, ReLU Activation, BatchNorm, and Dropout
    (except the last block).

    Attributes:
        conv_layers (nn.Sequential): Sequential container of convolutional layers,
            activation functions, batch normalization, and pooling.
        fc_out (nn.Linear): Fully connected output layer for final prediction.

    Examples:
        >>> model = CNN1D(input_dim=10, hidden_size=64, kernel_size=3, padding=1)
        >>> x = torch.randn(32, 20, 10)  # [batch_size, sequence_length, input_dim]
        >>> output = model(x)  # Shape: [32, 1]
    """

    def __init__(
        self, input_dim: int, hidden_size=64, kernel_size=3, padding=1, dropout=0.2
    ):
        """Initialize the 1D CNN model.

        Args:
            input_dim (int): Number of input features (channels)
            hidden_size (int, optional): Number of filters in the first conv layer. Defaults to 64.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
            padding (int, optional): Padding added to both sides of the input. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """

        super(CNN1D, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(
                self.input_dim,
                self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(self.dropout),
            # Second conv block
            nn.Conv1d(
                self.hidden_size,
                self.hidden_size // 2,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.Dropout(self.dropout),
            # Third conv block for better feature extraction
            nn.Conv1d(
                self.hidden_size // 2,
                16,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):
        """Perform the forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, 1)
        """

        x = x.permute(0, 2, 1)  # [batch_size, features, sequence_length]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc_out(x)
