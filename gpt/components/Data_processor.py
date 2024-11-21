import torch


class DataPreprocessor:
    """
    Handles data preprocessing: encoding text, decoding text, and generating training batches.

    Attributes:
        block_size (int): Length of the sequence block used for input and target.
        batch_size (int): Number of sequences in a batch.
        device (torch.device): Device to move data to (e.g., 'cpu' or 'cuda').
        text (str): Entire text loaded from the specified file.
        chars (list): Sorted list of unique characters in the text.
        stoi (dict): Mapping from characters to their integer indices.
        itos (dict): Mapping from integer indices to their characters.
        data (torch.Tensor): Encoded text data as a tensor of integer indices.
        train_data (torch.Tensor): Training split of the encoded data.
        val_data (torch.Tensor): Validation split of the encoded data.
    """

    def __init__(self, filepath, block_size, batch_size, device):
        """
        Initializes the DataPreprocessor by loading and processing the text file.

        Args:
            filepath (str): Path to the text file.
            block_size (int): Length of each sequence block.
            batch_size (int): Number of sequences in a batch.
            device (torch.device): Device to move tensors to (e.g., 'cpu' or 'cuda').
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Load text from file
        with open(filepath, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Extract unique characters and create encoding/decoding mappings
        self.chars = sorted(set(self.text))  # Sorted list of unique characters
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # Character to index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # Index to character

        # Encode text into integer indices
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

        # Split the data into training and validation sets (90% train, 10% val)
        split_idx = int(0.9 * len(self.data))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    def encode(self, text):
        """
        Encodes a string into a list of integers based on the character-to-index mapping.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of integers representing the encoded text.
        """
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        """
        Decodes a list of integers back into a string using the index-to-character mapping.

        Args:
            indices (list[int]): List of integer indices to decode.

        Returns:
            str: The decoded string.
        """
        return "".join([self.itos[i] for i in indices])

    def get_batch(self, split):
        """
        Generates a batch of input and target sequences from the specified data split.

        Args:
            split (str): Either "train" or "val" to specify which dataset to use.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x (torch.Tensor): Batch of input sequences of shape (batch_size, block_size).
                - y (torch.Tensor): Batch of target sequences of shape (batch_size, block_size).
        """
        # Choose data split (training or validation)
        data = self.train_data if split == "train" else self.val_data

        # Generate random starting indices for the batch
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # Create input and target tensors
        x = torch.stack(
            [data[i : i + self.block_size] for i in indices]
        )  # Input sequences
        y = torch.stack(
            [data[i + 1 : i + self.block_size + 1] for i in indices]
        )  # Target sequences

        # Move tensors to the specified device
        return x.to(self.device), y.to(self.device)
