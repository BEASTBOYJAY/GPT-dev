import torch.nn as nn
from components.Attention import MultiHeadAttention
from components.FeedFoward import FeedFoward


class Block(nn.Module):
    """
    A Transformer block that performs multi-head attention followed by a feed-forward network.

    The block consists of two main parts:
    1. Self-attention mechanism, which allows each token to attend to all other tokens.
    2. Feed-forward neural network, which applies further transformations to the input.

    Attributes:
        sa (MultiHeadAttention): Multi-head attention module for self-attention.
        ffwd (FeedFoward): Feed-forward module for further computation.
        ln1 (LayerNorm): Layer normalization applied after self-attention.
        ln2 (LayerNorm): Layer normalization applied after the feed-forward network.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initializes the Transformer block.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
            block_size (int): The size of the input sequence block.
            dropout (float): The dropout rate to apply during training.
        """
        super().__init__()

        # Calculate the size of each attention head
        head_size = n_embd // n_head

        # Initialize the multi-head attention layer
        self.sa = MultiHeadAttention(
            n_head, head_size, block_size=block_size, dropout=dropout, n_embd=n_embd
        )

        # Initialize the feed-forward network layer
        self.ffwd = FeedFoward(n_embd, dropout=dropout)

        # Initialize layer normalization layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size,T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: Output tensor after applying self-attention and feed-forward network.
        """
        # Apply self-attention and residual connection
        x = x + self.sa(self.ln1(x))

        # Apply feed-forward network and residual connection
        x = x + self.ffwd(self.ln2(x))

        return x
