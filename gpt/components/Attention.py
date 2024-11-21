import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """A single head of self-attention.

    This module performs scaled dot-product attention, including
    computing key, query, and value vectors, applying a causal mask,
    and returning the attention-weighted values.

    Attributes:
        n_embd (int): Dimensionality of the input embeddings.
        head_size (int): Size of the attention head.
        block_size (int): Maximum sequence length (used for masking).
        key (nn.Linear): Linear layer to compute key vectors.
        query (nn.Linear): Linear layer to compute query vectors.
        value (nn.Linear): Linear layer to compute value vectors.
        tril (torch.Tensor): Causal mask matrix.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        """
        Args:
            head_size (int): Size of each attention head.
            n_embd (int): Dimensionality of input embeddings.
            block_size (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size

        # Linear layers for projecting input to key, query, and value spaces.
        self.key = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias=False)

        # Register a causal mask to prevent attending to future positions.
        self.register_buffer(
            "tril", torch.tril(torch.ones(self.block_size, self.block_size))
        )

        # Dropout for regularization during training.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Performs forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time-steps, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time-steps, head_size).
        """
        # Extract dimensions from input.
        B, T, C = x.shape

        # Compute key, query, and value projections.
        k = self.key(x)  # Shape: (B, T, head_size)
        q = self.query(x)  # Shape: (B, T, head_size)

        # Compute scaled dot-product attention scores.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Shape: (B, T, T)

        # Apply causal masking to prevent attending to future positions.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # Shape: (B, T, T)

        # Apply softmax to normalize attention scores.
        wei = F.softmax(wei, dim=-1)  # Shape: (B, T, T)

        # Apply dropout for regularization.
        wei = self.dropout(wei)

        # Compute attention-weighted values.
        v = self.value(x)  # Shape: (B, T, head_size)
        out = wei @ v  # Shape: (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Combines multiple attention heads in parallel and projects the
    concatenated outputs to the original embedding dimension.

    Attributes:
        n_embd (int): Dimensionality of the input embeddings.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        block_size (int): Maximum sequence length.
        heads (nn.ModuleList): List of `Head` modules.
        proj (nn.Linear): Linear layer for projecting concatenated outputs.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        """
        Args:
            num_heads (int): Number of attention heads.
            head_size (int): Size of each attention head.
            n_embd (int): Dimensionality of input embeddings.
            block_size (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.n_embd = n_embd
        self.head_size = head_size
        self.num_heads = num_heads
        self.block_size = block_size

        # Create multiple attention heads.
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=self.head_size,
                    n_embd=self.n_embd,
                    block_size=self.block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        # Linear projection layer to combine outputs of all heads.
        self.proj = nn.Linear(head_size * num_heads, self.n_embd)

        # Dropout for regularization.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Performs forward pass of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time-steps, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time-steps, n_embd).
        """
        # Concatenate outputs of all attention heads along the last dimension.
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Project concatenated outputs back to embedding dimension.
        out = self.dropout(self.proj(out))

        return out
