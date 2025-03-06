import os
import math
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Custom attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output, attn_weights

# Feed-forward network used in transformer blocks
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x

# Custom transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Positional encoding for transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not model parameter but should be saved)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]

# Complete custom model architecture
class CustomModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer for prediction
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        # Create padding mask if not provided
        if mask is None:
            mask = (x != self.pad_idx).unsqueeze(-2)
        
        # Get embeddings and add positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def generate(self, start_tokens, max_length, temperature=1.0):
        """Generate sequence using the model with optional temperature sampling"""
        self.eval()
        device = next(self.parameters()).device
        current_tokens = start_tokens.clone().to(device)
        
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass
                logits = self(current_tokens)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to current tokens
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        return current_tokens

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            item,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Format for language modeling
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Shift for language modeling (predict next token)
        # Inputs are all tokens except the last, labels are all tokens except the first
        return {
            'input_ids': input_ids[:-1],
            'attention_mask': attention_mask[:-1],
            'labels': input_ids[1:]
        }

# Simple tokenizer for demonstration
class SimpleTokenizer:
    def __init__(self, vocab_file=None):
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        else:
            self.vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
            self.ids_to_tokens = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
    
    def add_tokens(self, tokens):
        """Add new tokens to the vocabulary"""
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[len(self.ids_to_tokens)] = token
    
    def encode(self, text):
        """Convert text to token IDs"""
        if isinstance(text, str):
            # Oversimplified tokenization - in real applications use subword tokenizers
            tokens = text.split()
            return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return []
    
    def decode(self, ids):
        """Convert token IDs to text"""
        return " ".join(self.ids_to_tokens.get(id, "<unk>") for id in ids if id != self.vocab["<pad>"])
    
    def encode_plus(self, text, max_length=512, padding='max_length', truncation=True, return_tensors=None):
        """Encode text with additional options similar to transformers"""
        encoded = self.encode(text)
        
        if truncation and len(encoded) > max_length - 2:  # Account for <sos> and <eos>
            encoded = encoded[:max_length - 2]
        
        # Add special tokens
        encoded = [self.vocab["<sos>"]] + encoded + [self.vocab["<eos>"]]
        
        # Add padding
        if padding == 'max_length':
            pad_length = max_length - len(encoded)
            if pad_length > 0:
                encoded = encoded + [self.vocab["<pad>"]] * pad_length
        
        # Create attention mask
        attention_mask = [1 if id != self.vocab["<pad>"] else 0 for id in encoded]
        
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([encoded]),
                'attention_mask': torch.tensor([attention_mask])
            }
        
        return {
            'input_ids': encoded,
            'attention_mask': attention_mask
        }
    
    def save_vocab(self, vocab_file):
        """Save vocabulary to file"""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    @property
    def vocab_size(self):
        return len(self.vocab)

# Training utilities
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a schedule with linear warmup and linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: Optional[LambdaLR] = None,
    clip_grad_norm: Optional[float] = None,
    use_amp: bool = False
):
    model.train()
    total_loss = 0
    scaler = GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        if use_amp:
            with autocast():
                outputs = model(input_ids, attention_mask)
                # Compute loss - reshape to [batch_size * seq_len, vocab_size]
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=model.pad_idx
                )
        else:
            outputs = model(input_ids, attention_mask)
            # Compute loss - reshape to [batch_size * seq_len, vocab_size]
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=model.pad_idx
            )
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Compute loss
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=model.pad_idx
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_model(model, tokenizer, output_dir, args=None):
    """Save model, tokenizer, and training args"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Save tokenizer vocabulary
    tokenizer.save_vocab(os.path.join(output_dir, "vocab.json"))
    
    # Save model config for future loading
    model_config = {
        "vocab_size": model.token_embedding.weight.size(0),
        "d_model": model.d_model,
        "num_layers": len(model.transformer_blocks),
        "num_heads": model.transformer_blocks[0].attention.num_heads,
        "d_ff": model.transformer_blocks[0].feedforward.linear1.out_features,
        "pad_idx": model.pad_idx,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save training args if provided
    if args is not None:
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

def load_model(model_dir, device=None):
    """Load a saved model and tokenizer"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # Initialize model with saved config
    model = CustomModel(**config)
    
    # Load state dict
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device)
    
    # Load tokenizer
    tokenizer = SimpleTokenizer(os.path.join(model_dir, "vocab.json"))
    
    return model, tokenizer

def train(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    logger.info("Loading data...")
    
    # This is a placeholder - in a real scenario, load your actual data
    # For demonstration, we'll create a dummy dataset
    if args.data_path:
        # Load real data
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f]
    else:
        # Create dummy data for demonstration
        data = [
            "this is a simple example sentence for training .",
            "machine learning models require data to learn patterns .",
            "custom architectures allow flexibility in model design .",
            "training neural networks is a complex process .",
            "transformers have revolutionized natural language processing ."
        ] * 20  # Repeat to have more data
    
    # Create and fit tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = SimpleTokenizer(args.vocab_file)
    
    # Update tokenizer vocabulary if using real data
    if not args.vocab_file:
        # Add words from the dataset to the vocabulary
        for text in data:
            tokens = text.split()
            tokenizer.add_tokens(tokens)
    
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = TextDataset(data, tokenizer, max_length=args.max_seq_len)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Initializing model...")
    model = CustomModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pad_idx=tokenizer.vocab["<pad>"]
    )
    model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    eval_losses = []
    best_eval_loss = float('inf')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            clip_grad_norm=args.max_grad_norm,
            use_amp=args.fp16
        )
        train_losses.append(train_loss)
        
        # Evaluate
        eval_loss = evaluate(model, eval_dataloader, device)
        eval_losses.append(eval_loss)
        
        logger.info(f"Train loss: {train_loss:.4f}, Eval loss: {eval_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        save_model(model, tokenizer, checkpoint_dir, args)
        
        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            logger.info(f"New best model with eval loss: {best_eval_loss:.4f}")
            best_model_dir = os.path.join(args.output_dir, "best_model")
            save_model(model, tokenizer, best_model_dir, args)
    
    # Save final model
    save_model(model, tokenizer, args.output_dir, args)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train loss')
    plt.plot(eval_losses, label='Eval loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_plot.png'))
    
    logger.info(f"Training complete. Model saved to {args.output_dir}")
    
    # Generate some sample text
    logger.info("Generating sample text...")
    sample_prompt = "this is"
    sample_input = torch.tensor([tokenizer.encode(sample_prompt)]).to(device)
    generated = model.generate(sample_input, max_length=20, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    
    logger.info(f"Sample prompt: '{sample_prompt}'")
    logger.info(f"Generated text: '{generated_text}'")
    
    # Return model and tokenizer
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a custom model architecture")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--vocab_file", type=str, default=None, help="Path to vocabulary file")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="custom_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
