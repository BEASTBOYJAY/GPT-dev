import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml


class DataProcessor:
    """
    Handles text processing, tokenization, and data preparation for training and validation.
    """

    def __init__(self, file_path: str, block_size: int, train_split: float = 0.9):
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = block_size

        # Encode text and split into train and validation datasets
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str):
        """Convert a string to a list of integers based on the vocabulary."""
        return [self.stoi[c] for c in s]

    def decode(self, l: list):
        """Convert a list of integers back to a string."""
        return "".join([self.itos[i] for i in l])

    def get_batch(self, split: str, batch_size: int, device: str):
        """
        Generate a batch of input-output pairs.
        """
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model with token embedding and sequence generation capabilities.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        B, T, C = logits.shape
        if targets is not None:
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to context
        return idx


class Trainer:
    """
    Manages training, evaluation, and optimization of the language model.
    """

    def __init__(
        self,
        model,
        data_processor,
        device,
        learning_rate,
        eval_iters,
        batch_size,
        epochs,
        eval_interval,
    ):
        self.model = model.to(device)
        self.data_processor = data_processor
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.eval_iters = eval_iters
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_interval = eval_interval

    def estimate_loss(self):
        """Estimate loss on training and validation datasets."""
        self.model.eval()
        losses = {split: torch.zeros(self.eval_iters) for split in ["train", "val"]}
        with torch.no_grad():
            for split in ["train", "val"]:
                for k in range(self.eval_iters):
                    X, Y = self.data_processor.get_batch(
                        split, self.batch_size, self.device
                    )
                    _, loss = self.model(X, Y)
                    losses[split][k] = loss.item()
        self.model.train()
        return {split: losses[split].mean().item() for split in ["train", "val"]}

    def train(self):
        """Train the model for a specified number of iterations."""
        for iter in range(self.epochs):
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

            xb, yb = self.data_processor.get_batch(
                "train", self.batch_size, self.device
            )
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()


class TrainandEval:
    def __init__(self, config_file: str):
        torch.manual_seed(1337)
        self.config = self.load_config(config_file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_processor = DataProcessor(**self.config["data_processor"])
        self.model = BigramLanguageModel(self.data_processor.vocab_size)
        self.trainer = Trainer(
            self.model,
            self.data_processor,
            **self.config["trainer"],
            device=self.device,
        )

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def train(self):
        self.trainer.train()

    def generate(
        self,
    ):
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        generated_text = self.data_processor.decode(
            self.model.generate(context, **self.config["generate"])[0].tolist()
        )

        return generated_text


if __name__ == "__main__":

    trainer = TrainandEval("config.yaml")
    trainer.train()
    print(trainer.generate())
