import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    """A simple Linear Regression model represnting a linear
    function y = wx + b, where w is weight vectore and b is 
    a bias vector. It uses PyTorch's nn.Linear module to
    implement transformation.

    Attributes:
        linear (nn.Linear): Linear layer to perform input_dim to 1 transformation.

    Examples:
        >>> model = LinearRegression(input_dim=3)
        >>> x = torch.randn(10, 3)  # 10 samples, 3 features each
        >>> output = model(x)  # Shape: [10, 1]
    """

    def __init__(self, input_dim: int):
        """Initialize the linear regression model.
        
        Args:
            input_dim (int): Dimentionality of input features
        """

        super(LinearRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(in_features=self.input_dim, out_features=1)

    def forward(self, x: torch.Tensor):
        """Perform the forward pass through the model

        Args:
            x (torch.Tensor): Input tensor shape (batch_size, input_dim_)

        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, 1)
        """
        
        return self.linear(x)
    