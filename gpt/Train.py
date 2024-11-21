import torch
import yaml
from components.Language_model import GPTLanguageModel
from components.Trainer import Trainer
from utils.utils import get_vocab_size


class GPTTrainerPipeline:
    """
    A pipeline for training and generating text using the GPT model.

    This class handles the entire process from loading the configuration,
    initializing the model and trainer, training the model, and generating text.
    """

    def __init__(self, config_path: str):
        """
        Initializes the GPTTrainerPipeline by loading the configuration and setting up the model and trainer.

        Args:
            config_path (str): The path to the configuration YAML file.
        """
        # Load configuration from YAML file
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Set device to CUDA if available, else use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load file path and vocabulary size from the configuration
        self.file_path = config["File"]["file_path"]
        self.vocab_size = get_vocab_size(self.file_path)

        # Training hyperparameters
        self.block_size = config["Training"]["block_size"]
        self.batch_size = config["Training"]["batch_size"]
        self.epochs = config["Training"]["epochs"]
        self.learning_rate = config["Training"]["learning_rate"]
        self.eval_iters = config["Training"]["eval_iters"]

        # Attention-related hyperparameters
        self.n_embd = config["Attention"]["n_embd"]
        self.n_head = config["Attention"]["n_head"]
        self.dropout = config["Attention"]["dropout"]

        # Transformer block-related hyperparameters
        self.n_layer = config["Transformer_block"]["n_layer"]

        # Model Save
        self.model_save = config["Model"]["model_save_path"]
        self.save_interval = config["Model"]["save_interval"]

        # Initialize the GPT model with the specified parameters and move it to the selected device
        self.model = GPTLanguageModel(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            dropout=self.dropout,
        ).to(self.device)

        # Initialize the Trainer class with the model and configuration
        self.trainer = Trainer(
            model=self.model,
            file_path=self.file_path,
            block_size=self.block_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            eval_iters=self.eval_iters,
            device=self.device,
            model_save=self.model_save,
            save_interval=self.save_interval,
        )

    def train_model(self):
        """
        Trains the model using the Trainer class.

        Returns:
            nn.Module: The trained GPT model.
        """
        return self.trainer.train()

    def generate_text(self, prompt: str, max_new_tokens=100):
        """
        Generates text from the trained model based on a given prompt.

        Args:
            prompt (str): The initial text to generate from.

        Returns:
            str: The generated text based on the prompt.
        """
        return self.trainer.generate_text(prompt, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    # Load configuration from the YAML file
    config_path = "gpt/confg.yaml"

    # Initialize the GPTTrainerPipeline with the given configuration
    pipeline = GPTTrainerPipeline(config_path=config_path)

    # Train the model using the pipeline
    trained_model = pipeline.train_model()

    # Generate text based on the prompt "The Lord of the Rings"
    generated_text = pipeline.generate_text("The Lord of the Rings")
    print(generated_text)
