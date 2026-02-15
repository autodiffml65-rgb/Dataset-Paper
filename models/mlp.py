import torch
import torch.nn as nn

class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) model i.e. feedforward neural network
    using PyTorch's nn.Module with adjustable layers, hidden sizes, and dropout.
    
    Attributes:
        layers (nn.Sequential): A Sequential container for all neuralk network layers.

    Examples:
        >>> model = MLP(input_dim=5, hidden_size=64, num_layers=2, dropout=0.2)
        >>> x = torch.randn(10, 5)  # 10 samples, 5 features each
        >>> output = model(x)  # Shape: [10, 1]
    """

    def __init__(self, input_dim: int, hidden_size=64, 
                 num_layers=2, dropout=0.2):
        """Initialize the MLP model.

        Args:
            input_dim (int): Number of input features.
            hidden_size (int, optional): Size of hidden layers. Defaults to 64.
            num_layers (int, optional): Number of hidden layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """

        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(self.input_dim, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.Dropout(self.dropout))
            if i + 1 == self.num_layers:
                layers.append(nn.Linear(self.hidden_size, 1))
                break
            self.input_dim = self.hidden_size
            self.hidden_size = self.hidden_size // 2
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        """Perform the forward pass through the model

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, 1)
        """
        
        return self.layers(x)
