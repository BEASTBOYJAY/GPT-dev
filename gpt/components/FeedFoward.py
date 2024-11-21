import torch.nn as nn


class FeedFoward(nn.Module):
    """
    A simple feedforward neural network consisting of a linear layer,
    a non-linear activation function, another linear layer, and dropout.

    This module is often used as the position-wise feedforward component
    in transformer architectures.

    Attributes:
        net (nn.Sequential): Sequential container for the feedforward layers,
            including two linear transformations, a ReLU activation, and a dropout layer.
    """

    def __init__(self, n_embd, dropout):
        """
        Initializes the FeedForward module.

        Args:
            n_embd (int): Dimensionality of the input embeddings.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()

        # Define a sequential feedforward network
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expands dimensionality by a factor of 4
            nn.ReLU(),  # Applies ReLU activation for non-linearity
            nn.Linear(
                4 * n_embd, n_embd
            ),  # Reduces back to the original dimensionality
            nn.Dropout(dropout),  # Applies dropout for regularization
        )

    def forward(self, x):
        """
        Applies the feedforward network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_embd),
            with the same shape as the input.
        """
        # Pass the input through the sequential network
        return self.net(x)
