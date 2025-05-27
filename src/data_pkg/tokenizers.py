
import json
from typing import List, Dict, Union

# Special tokens (can be expanded)
PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]" # Useful for 'cls' pooling strategy
UNK_TOKEN = "[UNK]" # For unknown k-mers if not building vocab dynamically

class KmerTokenizer:
    def __init__(self, k: int, stride: int, vocab: Union[Dict[str, int], str] = None, add_special_tokens: bool = True):
        self.k = k
        self.stride = stride
        self.add_special_tokens = add_special_tokens
        
        self.pad_token = PAD_TOKEN
        self.cls_token = CLS_TOKEN
        self.unk_token = UNK_TOKEN

        if vocab:
            if isinstance(vocab, str): # Path to vocab file
                with open(vocab, 'r') as f:
                    self.vocab = json.load(f)
            elif isinstance(vocab, dict):
                self.vocab = vocab
            else:
                raise ValueError("Vocab must be a dictionary or a path to a JSON file.")
        else:
            # Basic DNA alphabet + special tokens if no vocab provided (dynamic building would be more robust)
            print("Warning: No vocabulary provided. Using a minimal default or expecting dynamic building.")
            self.vocab = {self.pad_token: 0, self.cls_token: 1, self.unk_token: 2, 'A':3, 'C':4, 'G':5, 'T':6} # Minimal example

        if self.add_special_tokens:
            if self.cls_token not in self.vocab:
                self.vocab[self.cls_token] = len(self.vocab)
            if self.pad_token not in self.vocab:
                self.vocab[self.pad_token] = len(self.vocab)
            if self.unk_token not in self.vocab:
                 self.vocab[self.unk_token] = len(self.vocab)
        
        self.stoi = self.vocab
        self.itos = {i: s for s, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.stoi.get(self.pad_token, 0)
        self.cls_token_id = self.stoi.get(self.cls_token, 1)


    def _dna_to_kmers(self, sequence: str) -> List[str]:
        """Converts a DNA sequence into a list of k-mers."""
        kmers = []
        if not sequence or not isinstance(sequence, str):
            return []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmers.append(sequence[i:i+self.k])
        return kmers

    def encode(self, sequence: str, max_length: int = None) -> List[int]:
        """Encodes a DNA sequence into a list of token IDs."""
        kmers = self._dna_to_kmers(sequence)
        
        token_ids = []
        if self.add_special_tokens:
            token_ids.append(self.stoi[self.cls_token])

        for kmer in kmers:
            token_ids.append(self.stoi.get(kmer, self.stoi[self.unk_token])) # Use UNK for out-of-vocab k-mers

        if max_length:
            # Truncate if necessary (keeping CLS at the beginning)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            # Pad if necessary
            elif len(token_ids) < max_length:
                padding_needed = max_length - len(token_ids)
                token_ids.extend([self.stoi[self.pad_token]] * padding_needed)
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a k-mer string (or approximated DNA)."""
        # This basic decode might not perfectly reconstruct DNA if k-mers overlap
        # and special tokens are present. It's mostly for debugging.
        kmers = [self.itos.get(token_id, "") for token_id in token_ids if token_id != self.pad_token_id]
        return " ".join(kmers)

    @staticmethod
    def build_vocab_from_sequences(sequences: List[str], k: int, stride: int, min_freq: int = 1) -> Dict[str, int]:
        """Builds a k-mer vocabulary from a list of DNA sequences."""
        kmer_counts: Dict[str, int] = {}
        temp_tokenizer = KmerTokenizer(k, stride, vocab={}, add_special_tokens=False) # Temp for kmer generation

        for seq in sequences:
            kmers = temp_tokenizer._dna_to_kmers(seq)
            for kmer in kmers:
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
        
        vocab = {PAD_TOKEN: 0, CLS_TOKEN: 1, UNK_TOKEN: 2}
        idx = 3
        for kmer, count in kmer_counts.items():
            if count >= min_freq:
                vocab[kmer] = idx
                idx += 1
        print(f"Built vocabulary with {len(vocab)} k-mers (min_freq={min_freq}).")
        return vocab

    def save_vocab(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def from_pretrained(cls, vocab_path: str, k: int, stride: int, add_special_tokens: bool = True):
        with open(vocab_path, 'r') as f:
            vocab_dict = json.load(f)
        return cls(k=k, stride=stride, vocab=vocab_dict, add_special_tokens=add_special_tokens)

