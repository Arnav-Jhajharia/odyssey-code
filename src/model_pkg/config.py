# model_pkg/config.py
from dataclasses import dataclass, field

@dataclass
class DNAEncoderConfig:
    """
    Configuration for the DNASequenceEncoder model.
    """
    block_size: int = 256  # Max sequence length (context window)
    vocab_size: int = 512   # Placeholder, will be updated by tokenizer's actual vocab size
    n_layer: int = 6       # Number of Transformer encoder layers
    n_head: int = 6        # Number of attention heads
    n_embd: int = 384      # Embedding dimension
    dropout: float = 0.1   # Dropout rate
    bias: bool = True      # True to use bias in Linears and LayerNorms.
    # Add any other model-specific parameters here, e.g., pooling_strategy
    pooling_strategy: str = "cls" # 'cls' or 'mean'
