# train_dna_encoder.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import logging # For Hydra logging

# Setup basic logging for Hydra
log = logging.getLogger(__name__)


from model_pkg.config import DNAEncoderConfig
from model_pkg.dna_encoder import DNASequenceEncoder
from data_pkg.tokenizers import KmerTokenizer
from data_pkg.datasets import PairedDnaDataset
from training_pkg.losses import ContrastiveLoss # Or define it here

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch_num, log_every_n_steps):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        emb1 = model(input_ids1, attention_mask=attention_mask1)
        emb2 = model(input_ids2, attention_mask=attention_mask2)

        loss = loss_fn(emb1, emb2, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % log_every_n_steps == 0:
            log.info(f"Epoch {epoch_num+1}, Step {i+1}/{len(dataloader)}, Batch Loss: {loss.item():.4f}")
    
    avg_epoch_loss = total_loss / len(dataloader)
    log.info(f"Epoch {epoch_num+1} completed. Average Training Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)

            emb1 = model(input_ids1, attention_mask=attention_mask1)
            emb2 = model(input_ids2, attention_mask=attention_mask2)
            
            loss = loss_fn(emb1, emb2, labels)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(dataloader)
    log.info(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("--- Configuration ---")
    log.info(OmegaConf.to_yaml(cfg)) # Hydra logger will handle this
    log.info("---------------------")

    # --- Seed for reproducibility ---
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    # np.random.seed(cfg.seed) # if using numpy.random
    # random.seed(cfg.seed)   # if using random

    # --- Instantiate Tokenizer ---
    log.info("Instantiating tokenizer...")
    # Assuming vocab_file is defined in cfg.data and contains k-mers
    # Or build it dynamically if not provided (more complex setup)
    if not os.path.exists(cfg.data.vocab_file):
        log.error(f"Vocabulary file not found at {cfg.data.vocab_file}. Please create it first.")
        # Example: You might run a separate script to build vocab from your training data
        # df_train = pd.read_csv(cfg.data.data_path)
        # all_sequences = pd.concat([df_train['seq1'], df_train['seq2']]).dropna().unique().tolist()
        # vocab = KmerTokenizer.build_vocab_from_sequences(all_sequences, cfg.data.kmer_k, cfg.data.kmer_stride)
        # KmerTokenizer(k=cfg.data.kmer_k, stride=cfg.data.kmer_stride, vocab=vocab).save_vocab(cfg.data.vocab_file)
        # log.info(f"Attempted to build and save vocab to {cfg.data.vocab_file}. Please verify and re-run.")
        return

    tokenizer = KmerTokenizer.from_pretrained(
        vocab_path=cfg.data.vocab_file,
        k=cfg.data.kmer_k,
        stride=cfg.data.kmer_stride,
        add_special_tokens=True # Assuming CLS token for pooling
    )
    log.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # --- Instantiate Model ---
    log.info("Instantiating model...")
    # Update model config vocab_size based on tokenizer
    # The cfg.model.config is already an instance of DNAEncoderConfig
    model_config_obj: DNAEncoderConfig = cfg.model.config 
    model_config_obj.vocab_size = tokenizer.vocab_size # CRITICAL: Ensure model vocab matches tokenizer
    
    encoder_model = DNASequenceEncoder(config=model_config_obj)
    log.info(f"Model {type(encoder_model).__name__} instantiated successfully!")
    log.info(f"Number of parameters: {encoder_model.get_num_params():,}")

    # --- Setup Device ---
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")
    encoder_model.to(device)
    log.info(f"Model moved to device: {device}")

    # --- Prepare Datasets and DataLoaders ---
    log.info("Loading datasets...")
    train_dataset = PairedDnaDataset(
        csv_file=cfg.data.data_path,
        tokenizer=tokenizer,
        max_seq_length=model_config_obj.block_size # Use model's block_size
    )
    val_dataset = PairedDnaDataset(
        csv_file=cfg.data.val_data_path, # Ensure val_data_path is in your data config
        tokenizer=tokenizer,
        max_seq_length=model_config_obj.block_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    log.info(f"Training data: {len(train_dataset)} samples. Validation data: {len(val_dataset)} samples.")

    # --- Instantiate Optimizer and Loss Function ---
    if cfg.training.optimizer == "AdamW":
        optimizer = optim.AdamW(encoder_model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.get("adamw_weight_decay", 0.01))
    else: # Add other optimizers if needed
        log.error(f"Optimizer {cfg.training.optimizer} not supported.")
        return
    
    if cfg.training.loss_function == "ContrastiveLoss":
        loss_fn = ContrastiveLoss(margin=cfg.training.contrastive_margin)
    else: # Add other loss functions
        log.error(f"Loss function {cfg.training.loss_function} not supported.")
        return
    log.info(f"Optimizer: {cfg.training.optimizer}, Loss Function: {cfg.training.loss_function}")

    # --- Training Loop ---
    log.info("Starting training...")
    best_val_loss = float('inf')
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # Get Hydra's output dir
    checkpoint_dir = os.path.join(output_dir, cfg.training.checkpoint_dir_leafname) # Use a leafname from config
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.training.num_epochs):
        log.info(f"--- Epoch {epoch+1}/{cfg.training.num_epochs} ---")
        train_loss = train_epoch(encoder_model, train_dataloader, optimizer, loss_fn, device, epoch, cfg.training.log_every_n_steps)
        val_loss = validate_epoch(encoder_model, val_dataloader, loss_fn, device)

        # WandB Logging (if enabled in config)
        if cfg.training.logging.use_wandb:
            try:
                import wandb
                # Initialize wandb if it's the first epoch and not already initialized
                if epoch == 0 and wandb.run is None: # Check if wandb.run is None
                    wandb.init(
                        project=cfg.training.logging.wandb_project,
                        entity=cfg.training.logging.wandb_entity,
                        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), # Log Hydra config
                        name=f"{cfg.model.name}-run-{output_dir.split('/')[-1]}", # Example run name
                        dir=output_dir # Save wandb files in Hydra output dir
                    )
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]['lr']})
            except ImportError:
                log.warning("wandb not installed, skipping wandb logging. To use, pip install wandb")
            except Exception as e:
                log.error(f"Error during wandb logging: {e}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(encoder_model.state_dict(), best_model_path)
            log.info(f"New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")

        if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
            epoch_model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(encoder_model.state_dict(), epoch_model_path)
            log.info(f"Checkpoint saved to {epoch_model_path}")
            
    log.info("Training complete.")
    log.info(f"Best validation loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints saved in: {checkpoint_dir}")

    if cfg.training.logging.use_wandb and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
