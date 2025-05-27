# data_pkg/datasets.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict # Removed List, Tuple as they are not used in type hints here
from .tokenizers import KmerTokenizer # Assuming tokenizers.py is in the same package

class PairedDnaDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: KmerTokenizer, max_seq_length: int):
        """
        Args:
            csv_file (string): Path to the csv file with 'seq1', 'seq2', 'label'.
            tokenizer (KmerTokenizer): KmerTokenizer instance.
            max_seq_length (int): Maximum sequence length for padding/truncation.
                                  This should typically be model.config.block_size.
        """
        try:
            self.data_frame = pd.read_csv(csv_file)
            print(f"Successfully loaded {len(self.data_frame)} rows from {csv_file}")
        except FileNotFoundError:
            print(f"ERROR: CSV file not found at {csv_file}")
            raise
        except pd.errors.EmptyDataError:
            print(f"ERROR: CSV file at {csv_file} is empty.")
            raise
        except Exception as e:
            print(f"ERROR: Could not read CSV file at {csv_file}. Error: {e}")
            raise
        
        if not all(col in self.data_frame.columns for col in ['seq1', 'seq2', 'label']):
            raise ValueError("CSV file must contain 'seq1', 'seq2', and 'label' columns.")
            
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx): # Handle tensor indices if any
            idx = idx.item()

        row = self.data_frame.iloc[idx]
        seq1_str = str(row['seq1'])
        seq2_str = str(row['seq2'])
        label = float(row['label']) # ContrastiveLoss typically expects float (0.0 or 1.0)

        # Tokenize and pad/truncate
        encoded_seq1 = self.tokenizer.encode(seq1_str, max_length=self.max_seq_length)
        encoded_seq2 = self.tokenizer.encode(seq2_str, max_length=self.max_seq_length)

        # Create attention masks (1 for real tokens, 0 for padding)
        # This is crucial for the Transformer to ignore padding.
        attention_mask1 = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in encoded_seq1]
        attention_mask2 = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in encoded_seq2]
        
        # Ensure all lists are of max_seq_length (already handled by tokenizer.encode if max_length is passed)
        # assert len(encoded_seq1) == self.max_seq_length, f"Seq1 encoding length mismatch: {len(encoded_seq1)} vs {self.max_seq_length}"
        # assert len(attention_mask1) == self.max_seq_length, f"Seq1 mask length mismatch: {len(attention_mask1)} vs {self.max_seq_length}"
        # assert len(encoded_seq2) == self.max_seq_length, f"Seq2 encoding length mismatch: {len(encoded_seq2)} vs {self.max_seq_length}"
        # assert len(attention_mask2) == self.max_seq_length, f"Seq2 mask length mismatch: {len(attention_mask2)} vs {self.max_seq_length}"


        sample = {
            'input_ids1': torch.tensor(encoded_seq1, dtype=torch.long),
            'attention_mask1': torch.tensor(attention_mask1, dtype=torch.long),
            'input_ids2': torch.tensor(encoded_seq2, dtype=torch.long),
            'attention_mask2': torch.tensor(attention_mask2, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }
        return sample

# Utility function to build and save vocabulary (run once or integrate into a prep script)
# To be called from a separate script or your main training script if vocab doesn't exist.
def build_and_save_vocab_if_needed(
    data_paths: List[str], 
    k: int, 
    stride: int, 
    vocab_output_path: str, 
    min_freq: int = 1,
    force_rebuild: bool = False):
    """
    Builds a Kmer vocabulary from sequences in specified CSV files and saves it.
    Only builds if the vocab file doesn't exist or if force_rebuild is True.
    """
    if os.path.exists(vocab_output_path) and not force_rebuild:
        print(f"Vocabulary already exists at {vocab_output_path}. Skipping build.")
        return

    print(f"Building vocabulary for k={k}, stride={stride}...")
    all_sequences = set() # Use a set to store unique sequences
    for data_path in data_paths:
        try:
            df = pd.read_csv(data_path)
            for seq_col in ['seq1', 'seq2']:
                if seq_col in df.columns:
                    all_sequences.update(df[seq_col].dropna().astype(str).tolist())
        except FileNotFoundError:
            print(f"Warning: Data file {data_path} not found during vocab building.")
        except Exception as e:
            print(f"Warning: Error reading {data_path} during vocab building: {e}")
    
    if not all_sequences:
        print("No sequences found to build vocabulary. Please check data paths.")
        return

    print(f"Found {len(all_sequences)} unique sequences for vocab building.")
    vocab_dict = KmerTokenizer.build_vocab_from_sequences(list(all_sequences), k, stride, min_freq=min_freq)
    
    # Ensure parent directory exists for vocab_output_path
    os.makedirs(os.path.dirname(vocab_output_path), exist_ok=True)
    with open(vocab_output_path, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"Vocabulary saved to {vocab_output_path} with {len(vocab_dict)} tokens.")

