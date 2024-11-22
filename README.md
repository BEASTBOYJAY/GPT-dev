# GPT Language Model Implementation

A PyTorch-based implementation of a GPT-style language model featuring multi-head attention and a transformer architecture. This project offers a comprehensive pipeline for training and deploying a GPT model for text generation, utilizing character-level tokenization..

## Features

- **Modular Architecture**: Clean separation of model components (attention, feed-forward, transformer blocks)
- **Configurable Model**: Easy hyperparameter tuning through YAML configuration
- **Training Pipeline**: Complete training infrastructure with progress tracking
- **Inference Support**: Dedicated pipeline for text generation using trained models
- **GPU Acceleration**: Automatic GPU utilization when available
- **Checkpoint Management**: Regular model saving and loading capabilities

## Project Structure

```
├── Notebooks/                   # Different .pynb files used for research/learning  
├── bigram_model/
│   ├── bigram_model.py          # Simple bigram model which is a inital for learning the gpt model 
│   ├── config.yaml              # Config file for the trainig and inference of the bigram model
│
├── gpt/
│   ├── components/
│   │   ├── Attention.py         # Single and multi-head attention implementations
│   │   ├── FeedForward.py       # Feed-forward network component
│   │   ├── Transformer_block.py # Transformer block implementation
│   │   ├── Language_model.py    # Main GPT model architecture
│   │   ├── Data_processor.py    # Data handling and preprocessing
│   │   └── Trainer.py           # Training loop and utilities
│   ├── utils/
│   │   └── utils.py             # Utility functions
│   ├── Train.py                 # Training pipeline
│   ├── Inference.py             # Inference pipeline
│   └── config.yaml              # Model and training configuration
│
│
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BEASTBOYJAY/GPT-dev.git
cd GPT-dev
```

2. Create an Virtual Enviroment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The model and training parameters can be configured in `config.yaml`:

```yaml
Attention:
  n_embd: 128      # Embedding dimension
  n_head: 8        # Number of attention heads
  dropout: 0.1     # Dropout rate

Transformer_block:
  n_layer: 4       # Number of transformer layers

Training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  eval_iters: 500
  block_size: 128  # Maximum sequence length

File:
  file_path: "input.txt"  # Path to training data

Model_save:
  model_save_path: "results"
  save_interval: 2        # Save every N epochs
```

## Usage

### Training

1. Prepare your training data in a text file and update the `file_path` in `config.yaml`.

2. Run the training pipeline:
```python
from Train import GPTTrainerPipeline

# Initialize and train the model
pipeline = GPTTrainerPipeline(config_path="config.yaml")
trained_model = pipeline.train_model()

# Generate sample text
generated_text = pipeline.generate_text()
print(generated_text)
```

### Inference

To use a trained model for text generation:

```python
from Inference import GPTInferencePipeline

# Initialize the pipeline
pipeline = GPTInferencePipeline(config_path="config.yaml")

# Load a trained model
pipeline.load_model("results/model_epoch_400.pt")

# Generate text
generated_text = pipeline.generate_text(prompt="Your prompt here")
print(generated_text)
```

## Model Architecture

The implementation follows the standard GPT architecture:

1. **Token Embeddings**: Convert input tokens to continuous vectors
2. **Positional Embeddings**: Add position information to token embeddings
3. **Transformer Blocks**: Multiple layers of:
   - Multi-head self-attention
   - Feed-forward neural network
   - Layer normalization
   - Residual connections
4. **Language Model Head**: Final projection to vocabulary size

### Attention Mechanism

The attention mechanism includes:
- Scaled dot-product attention
- Multi-head attention with parallel attention heads
- Causal masking for autoregressive generation

## Training Process

The training pipeline includes:
- Batch-wise training with cross-entropy loss
- Regular evaluation on validation set
- Progress tracking with tqdm
- Periodic model checkpointing
- GPU acceleration when available

## Acknowledgments

This implementation is inspired by the GPT architecture as described in the OpenAI papers and various open-source implementations.
