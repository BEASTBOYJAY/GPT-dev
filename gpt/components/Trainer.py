import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from components.Data_processor import DataPreprocessor
from tqdm import tqdm


class Trainer:
    """
    A class for training and evaluating a GPT model.

    Attributes:
        model (nn.Module): The GPT model to train.
        optimizer (optim.Optimizer): The optimizer for training (AdamW).
        device (torch.device): The device to train the model on (CPU or GPU).
        eval_iters (int): The number of iterations to run when estimating loss.
        epochs (int): The number of epochs to train for.
        save_interval (int): The interval (in epochs) at which to save the model.
        data_processor (DataPreprocessor): Handles data preprocessing, batching, and encoding.
    """

    def __init__(
        self,
        model,
        file_path,
        block_size,
        batch_size,
        learning_rate,
        device,
        eval_iters,
        epochs,
        model_save,
        save_interval=10,
    ):
        """
        Initializes the Trainer with the specified parameters.

        Args:
            model (nn.Module): The GPT model to train.
            file_path (str): Path to the dataset file.
            block_size (int): Sequence length for each input example.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            device (torch.device): The device to train the model on (CPU or GPU).
            eval_iters (int): Number of iterations for loss estimation.
            epochs (int): Number of epochs to train the model for.
        """
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.device = device
        self.eval_iters = eval_iters
        self.epochs = epochs
        self.save_interval = save_interval
        self.model_save = model_save

        # Initialize data processor for encoding, batching, and splitting data
        self.data_processor = DataPreprocessor(
            filepath=file_path,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
        )

    @torch.no_grad()
    def estimate_loss(self):
        """
        Estimate the loss on the train and validation sets.

        This method runs in evaluation mode and computes the loss for both
        training and validation datasets over a specified number of iterations.

        Returns:
            dict: A dictionary containing the average loss for both 'train' and 'val' splits.
        """
        out = {}
        self.model.eval()  # Set model to evaluation mode
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                # Get batch of data
                X, Y = self.data_processor.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()  # Average loss for the split
        self.model.train()  # Set model back to training mode
        return out

    def train_step(self):
        """
        Perform a single training step.

        This method processes a batch of data, computes the loss, and updates the model's weights.

        Returns:
            float: The loss value for the current training step.
        """
        # Get a batch of training data
        xb, yb = self.data_processor.get_batch("train")

        # Compute loss
        logits, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)  # Clear previous gradients
        loss.backward()  # Backpropagate gradients
        self.optimizer.step()  # Update the model's weights

        return loss.item()  # Return the loss for logging

    def train(self):
        """
        Complete the training process over multiple epochs with enhanced progress tracking.

        This method trains the model for the specified number of epochs and provides
        a progress bar with detailed loss information.

        Returns:
            nn.Module: The trained GPT model.
        """

        # Create a main progress bar for epochs
        epoch_pbar = tqdm(range(self.epochs), desc="Training Progress", position=0)

        for iter in epoch_pbar:
            # Estimate loss on train and validation datasets
            losses = self.estimate_loss()

            # Update the epoch progress bar with current loss information
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f'{losses["train"]:.4f}',
                    "Val Loss": f'{losses["val"]:.4f}',
                }
            )

            # Perform a training step
            loss = self.train_step()

            if iter % self.save_interval == 0:
                self.save_model(iter)

        return self.model

    def save_model(self, epoch):
        # Save the model at the specified epoch
        checkpoint_path = f"{self.model_save}/model_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    def generate_text(self, context=None, max_new_tokens=100):
        """
        Generate text from the trained model given a context.

        Args:
            context (str, optional): The initial context (prompt) to start text generation.
                Defaults to None, which initializes with an empty context.
            max_new_tokens (int): The number of new tokens to generate. Defaults to 100.

        Returns:
            str: The generated text as a string.
        """
        if context is None:
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            # Encode the provided context string into token indices
            encoded_context = self.data_processor.encode(context)
            context = torch.tensor(
                encoded_context, dtype=torch.long, device=self.device
            ).unsqueeze(0)

        # Ensure context is on the correct device (CPU or GPU)
        context = context.to(self.device)

        # Generate tokens using the model's generate method
        generated_tokens = self.model.generate(context, max_new_tokens=max_new_tokens)

        # Decode the generated token indices back into text
        generated_text = self.data_processor.decode(generated_tokens[0].tolist())
        return generated_text
