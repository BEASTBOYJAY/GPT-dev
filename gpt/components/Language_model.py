import torch
import torch.nn as nn
from torch.nn import functional as F
from components.Transformer_block import Block


class GPTLanguageModel(nn.Module):
    """
    A Generative Pre-trained Transformer (GPT) language model for next-token prediction.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        n_embd (int): Dimensionality of token embeddings.
        block_size (int): Maximum sequence length the model can process.
        n_head (int): Number of attention heads.
        n_layer (int): Number of transformer blocks in the model.
        dropout (float): Dropout probability for regularization.
        token_embedding_table (nn.Embedding): Embedding layer for token indices.
        position_embedding_table (nn.Embedding): Embedding layer for positional indices.
        blocks (nn.Sequential): Stack of transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        lm_head (nn.Linear): Linear layer projecting to vocabulary size for logits.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        """
        Initializes the GPTLanguageModel with the specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            n_embd (int): Dimensionality of embeddings.
            block_size (int): Maximum sequence length the model can process.
            n_head (int): Number of attention heads.
            n_layer (int): Number of transformer blocks.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        # Token and position embedding layers
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    self.n_embd,
                    block_size=self.block_size,
                    n_head=self.n_head,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layer)
            ]
        )

        # Final layer normalization and projection
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for linear and embedding layers.

        Args:
            module (nn.Module): Module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T), where B is the batch size
                and T is the sequence length.
            targets (torch.Tensor, optional): Target tensor of shape (B, T) for computing
                cross-entropy loss. Defaults to None.

        Returns:
            tuple:
                - logits (torch.Tensor): Logits of shape (B, T, vocab_size).
                - loss (torch.Tensor or None): Cross-entropy loss if targets are provided.
        """
        B, T = idx.shape

        # Token and positional embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)
        x = tok_emb + pos_emb  # Combine token and positional embeddings (B, T, n_embd)

        # Pass through transformer blocks and final normalization
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)

        # Project to vocabulary size for logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Flatten logits
            targets = targets.view(B * T)  # Flatten targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation of new tokens given a context.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T), where B is the batch size
                and T is the current sequence length.
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            torch.Tensor: Tensor of shape (B, T + max_new_tokens) with the generated sequence.
        """
        for _ in range(max_new_tokens):
            # Focus on the last block_size tokens for prediction
            idx_cond = idx[:, -self.block_size :]

            # Compute logits and get predictions
            logits, _ = self(idx_cond)
            logits = logits[
                :, -1, :
            ]  # Extract logits for the last time step (B, vocab_size)

            # Convert logits to probabilities and sample next token
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample (B, 1)

            # Append the sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx
