# model_pkg/dna_encoder.py
import torch
import torch.nn as nn
import math
from .encoder_components import EncoderBlock, LayerNorm # Assuming components are in the same package
from .config import DNAEncoderConfig # Import the config dataclass

class DNASequenceEncoder(nn.Module):
    def __init__(self, config: DNAEncoderConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), # Max sequence length
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config.n_embd, config.n_head, config.dropout, config.bias)
                               for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.output_embedding_dim = config.n_embd

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, similar to GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): # In MLP and Attention output projections
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"DNASequenceEncoder initialized with {self.get_num_params():,} parameters. Vocab size: {config.vocab_size}")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """ Returns the number of parameters in the model. """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude positional embeddings if non_embedding is True
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        """ Initializes weights of linear and embedding layers. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DNASequenceEncoder.
        Args:
            input_ids (torch.Tensor): Tensor of token IDs, shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Boolean tensor indicating padding,
                                                    shape (batch_size, seq_len).
                                                    1 for real tokens, 0 for padding.
        Returns:
            torch.Tensor: Sequence embedding, shape (batch_size, n_embd).
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Forward the Transformer model
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)       # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # The attention_mask for EncoderBlock should be (B, T_key) where 1 is keep, 0 is pad.
        # The MultiHeadSelfAttention will handle converting it for flash/manual attention.
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask) # Pass (B, T) mask

        x = self.transformer.ln_f(x) # (b, t, n_embd)

        # Pooling strategy
        if self.config.pooling_strategy == 'cls':
            # Assumes the first token (index 0) is the CLS token
            sequence_embedding = x[:, 0] # (b, n_embd)
        elif self.config.pooling_strategy == 'mean':
            if attention_mask is not None:
                # Only mean pool over non-padded tokens
                # Expand attention_mask to match embedding dimensions for masking
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
                sum_embeddings = torch.sum(x * input_mask_expanded, dim=1)
                # Count non-padded tokens for each sequence in the batch
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                sequence_embedding = sum_embeddings / sum_mask
            else:
                # If no attention mask, mean pool all token embeddings
                sequence_embedding = torch.mean(x, dim=1) # (b, n_embd)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.config.pooling_strategy}")

        return sequence_embedding # (b, n_embd)
