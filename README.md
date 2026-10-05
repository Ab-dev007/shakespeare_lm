# Shakespeare Language Model

A character-level language model built from scratch in PyTorch, progressing from a bigram baseline to a GPT-style transformer trained on Tiny Shakespeare.

**Python · PyTorch · Causal self-attention · Autoregressive generation**

This learning project implements the data pipeline, model components, and training loop directly, making it possible to follow the path from raw text to generated characters.

## Reported results

| Model | Parameters | Validation loss | Sample output |
| --- | ---: | ---: | --- |
| Bigram | 4,225 | ~2.4 | `HERDms t of IXf hasen d` |
| Transformer | 209,729 | ~1.75 | `WARWICK: Now how their come, our chown` |

These are approximate results recorded for this project, rather than a standardized benchmark. The sample illustrates emerging dialogue formatting and character names; the generated text still has substantial grammatical and semantic limitations.

## Architecture

The transformer uses:

- Learned character and positional embeddings
- Causal multi-head self-attention
- Position-wise feed-forward layers with 4× expansion
- Residual connections and pre-norm layer normalization
- Dropout for regularization

The training workflow includes validation, gradient clipping, and checkpointing. A bigram model provides a simple baseline for comparison.

### Documented configuration

```python
vocab_size = 65
block_size = 32
d_model = 64
num_heads = 4
num_layers = 4
dropout = 0.2
batch_size = 32
learning_rate = 1e-3
training_steps = 5000
```

## Quick start

Prerequisites: Git, Conda or Miniforge, and a compatible PyTorch installation. The original project setup targets Apple Silicon, with CPU/CUDA use depending on the device configuration in the training code.

```bash
git clone https://github.com/MindForge-Abhishek/shakespeare_lm.git
cd shakespeare_lm

conda env create -f environment.yml
conda activate shakespeare_lm
pip install torch
```

### Prepare the data

Run commands from the repository root:

```bash
python -m src.data.download
python -m src.data.tensor_creation
```

### Train the models

```bash
# Bigram baseline
python -m src.training.train

# Transformer
python -m src.training.transformer_training
```

The documented transformer checkpoint path is `checkpoints/transformer.pt`.

## Project structure

```text
shakespeare_lm/
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── tokenizer.py
│   │   └── tensor_creation.py
│   ├── models/
│   │   ├── bigram.py
│   │   └── transformer.py
│   └── training/
│       ├── data_loader.py
│       ├── train.py
│       └── transformer_training.py
├── notebooks/           # Data exploration and attention experiments
├── data/                # Generated locally; excluded from version control
│   ├── raw/
│   └── processed/
├── checkpoints/         # Generated model weights; excluded from version control
├── environment.yml
└── README.md
```

## What I explored

| Area | Concepts |
| --- | --- |
| Data | Character-level vocabulary, encoding/decoding, tensors, train/validation splitting |
| Modeling | Bigram baseline, attention, causal masking, positional embeddings, transformer blocks |
| Optimization | Cross-entropy loss, backpropagation, dropout, gradient clipping |
| Training workflow | Batch sampling, validation, checkpointing |
| Generation | Autoregressive sampling and context cropping |

## Dataset and scope

The project uses [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), a small Shakespeare text corpus containing 1,115,394 characters and 65 unique characters in the documented dataset. It is not the complete works of Shakespeare.

This is an educational model with a short context window and a small training corpus. Its results demonstrate the implementation and learning process, not general-purpose language understanding or production readiness.

## Acknowledgements

The transformer architecture builds on [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). The dataset is distributed through Andrej Karpathy's [char-rnn repository](https://github.com/karpathy/char-rnn).
