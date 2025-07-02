import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
from ethan_decoder_model import DecoderFull
from ethan_flickr_dataset_processing import collate_fn, convert_to_tensors
from token_utils import TokenConverter
from dotenv import load_dotenv
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

load_dotenv(override=True)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "")


class FlickrDataset(Dataset):
    """Custom dataset class for Flickr30k processed data."""
    
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Check if text_tokens exists, if not create it from text_embed
        if 'text_tokens' not in item:
            # This is an old dataset without text_tokens
            # We need to regenerate tokens from the caption
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            tokens = tokenizer.encode(item['caption'], max_length=128, truncation=True, padding='max_length')
            item['text_tokens'] = tokens.tolist()
        
        # Convert to tensors
        item_tensors = convert_to_tensors(item)
        return item_tensors


def prepare_input_target_sequences(batch, device, num_image_patches=50):
    image_embeds = batch["image_embeds"].to(device)  
    text_embeds = batch["text_embeds"].to(device)    
    text_tokens = batch["text_tokens"].to(device)    
    
    # Token IDs
    eos_token_id = 2
    bos_token_id = 0
    pad_token_id = 1
    
    # Keep input text embeddings as-is (no modifications needed)
    input_text_embeds = text_embeds
    
    # Create target tokens for the entire sequence (image + text)
    batch_size, text_seq_len = text_tokens.shape
    target_tokens = torch.full((batch_size, 50 + text_seq_len), -100, device=device)  # Start with all ignore
    
    # Set text portion (shift text tokens left by 1 to remove BOS)
    target_tokens[:, 50:] = text_tokens  # Keep original text tokens
    target_tokens[:, 50] = -100  # Ignore the BOS token position
    
    # Set padding tokens to ignore
    target_tokens[target_tokens == pad_token_id] = -100
    
    # Concatenate image and text embeddings
    input_sequence = torch.cat([image_embeds, input_text_embeds], dim=1)
    
    return input_sequence, target_tokens

def train_one_epoch(model, dataloader, optimizer, criterion, device, token_converter):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Prepare sequences
        input_sequence, target_tokens = prepare_input_target_sequences(batch, device)
        
        # Forward pass
        logits = model(input_sequence)  # [batch_size, seq_len, vocab_size]
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_tokens_flat = target_tokens.reshape(-1)
        
        # Calculate loss (automatically ignores -100 tokens)
        loss = criterion(logits_flat, target_tokens_flat)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        wandb.log({"batch_loss": loss.item()})
    
    return total_loss / num_batches


def evaluate_model(model, dataloader, device, token_converter, save_dir="eval_results"):
    """Evaluate the model and save visualization."""
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_losses = []
    evaluation_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Prepare sequences
            input_sequence, target_tokens = prepare_input_target_sequences(batch, device)
            
            # Forward pass
            logits = model(input_sequence)
            
            # Use full sequence for evaluation (consistent with training)
            criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            batch_size, seq_len, vocab_size = logits.shape
            
            for i in range(batch_size):
                sample_logits = logits[i]  # [full_seq_len, vocab_size] 
                sample_targets = target_tokens[i]  # [full_seq_len]
                
                # Calculate loss using full sequence
                sample_logits_flat = sample_logits.reshape(-1, sample_logits.shape[-1])
                sample_targets_flat = sample_targets.reshape(-1)
                loss = criterion(sample_logits_flat, sample_targets_flat)
                
                # Get text portion for prediction display
                num_image_patches = 50
                text_logits = sample_logits[num_image_patches:]  # Text portion only
                
                # Store sample info
                evaluation_samples.append({
                    'loss': loss.mean().item(),  # Average loss for this sample
                    'image': batch['images'][i],
                    'caption': batch['captions'][i],
                    'predicted_tokens': torch.argmax(text_logits, dim=-1),
                    'target_tokens': sample_targets[num_image_patches:]  # Text portion only
                })
                
                all_losses.append(loss.mean().item())
    
    # Sort by loss for top/bottom analysis
    evaluation_samples.sort(key=lambda x: x['loss'])
    
    # Get top 3, bottom 3, and 3 random samples
    top_3 = evaluation_samples[:3]
    bottom_3 = evaluation_samples[-3:]
    
    # Fixed:
    if len(evaluation_samples) > 6:
        random_3 = random.sample(evaluation_samples[3:-3], min(3, len(evaluation_samples) - 6))
    else:
        random_3 = evaluation_samples[:min(3, len(evaluation_samples))]
    
    # Create visualizations
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    sample_sets = [
        (top_3, "Top 3 (Lowest Loss)"),
        (bottom_3, "Bottom 3 (Highest Loss)"),
        (random_3, "Random 3")
    ]
    
    for row, (samples, title) in enumerate(sample_sets):
        for col, sample in enumerate(samples):
            ax = axes[row, col]
            
            # Display image
            if isinstance(sample['image'], torch.Tensor):
                image = sample['image'].cpu().numpy()
                # Handle different tensor formats
                if image.shape[0] == 3:  # Channel first
                    image = image.transpose(1, 2, 0)
                # Normalize if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            else:
                image = sample['image']

            ax.imshow(image)
            ax.axis('off')
            
            # Convert tokens to text
            actual_caption = sample['caption']
            
            # Convert predicted tokens to text (only non-padded tokens)
            pred_tokens = sample['predicted_tokens']
            
            try:
                predicted_caption = token_converter.tokens_to_text_direct(
                    pred_tokens, skip_special_tokens=True
                )
            except:
                predicted_caption = "Error in decoding"
            
            # Set title with captions
            title_text = f"Loss: {sample['loss']:.3f}\nActual: {actual_caption[:30]}...\nPredicted: {predicted_caption[:30]}..."
            ax.set_title(title_text, fontsize=8)
        
        # Add row title
        fig.text(0.02, 0.83 - row * 0.28, title, rotation=90, fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    avg_loss = np.mean(all_losses)
    
    # Placeholder for other metrics
    metrics = {
        'avg_loss': avg_loss,
        'num_samples': len(evaluation_samples),
        # TODO: Add BLEU, perplexity, token accuracy here
    }
    
    return metrics


def train(model, train_dataset, val_dataset, config, device):
    """Main training function."""
    
    # Initialize token converter for evaluation
    token_converter = TokenConverter("roberta-base")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # To this:
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    
    print(f"Starting training for {config.num_epochs} epoch(s)...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, token_converter)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        print("Running evaluation...")
        eval_metrics = evaluate_model(model, val_loader, device, token_converter)
        print(f"Validation Loss: {eval_metrics['avg_loss']:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": eval_metrics['avg_loss']
        })
        
        # Save model checkpoint
        checkpoint_path = f"model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': eval_metrics['avg_loss'],
            'config': config
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    
    return model, eval_metrics


def main():
    # Initialize Weights & Biases
    # Get vocabulary size from tokenizer
    temp_converter = TokenConverter("roberta-base")
    temp_converter.load_tokenizer()
    vocab_size = temp_converter.tokenizer.vocab_size
    

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load pre-embedded dataset
    data_path = 'flickr30k_processed_splits_current.pkl'  # Use the main file, not test
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'flickr30k_processed_splits.pkl')
    
    with open(data_path, 'rb') as f:
        raw_dataset = pickle.load(f)

    # Check if the dataset is loaded correctly
    if not raw_dataset:
        raise ValueError("Dataset is empty or not loaded correctly.")
    
    # Get splits
    train_data = raw_dataset.get('train', [])
    validation_data = raw_dataset.get('validation', [])
    test_data = raw_dataset.get('test', [])

        # Get text token length from first item (assuming all are same length)
    sample_item = train_data[0]
    if 'text_tokens' in sample_item:
        max_text_length = len(sample_item['text_tokens'])
    else:
        max_text_length = 60  # fallback
    print(f"Text token length: {max_text_length}")
    calculated_seq_len = 50 + max_text_length   # 49 image patches + text tokens
    
    wandb.init(
        project=WANDB_PROJECT or "ethan_decoder_training", 
        entity=WANDB_ENTITY,
        config={
            "seq_len": calculated_seq_len,  # Instead of 110
            "dim_attn_in": 768,  # RoBERTa hidden size
            "num_decoder_blocks": 1,
            "dim_ffn": 2048,
            "num_msAttnHeads": 1,
            "dim_V": 64,
            "dim_KQ": 64,
            "dim_logits": vocab_size,  # Use actual vocabulary size
            "dropout_rate": 0.1,
            "batch_size": 1,  # Reduced for memory
            "num_epochs": 1,
            "learning_rate": 0.0001
        }
    )
    # Create the model
    model = DecoderFull(wandb.config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")


    # Check if the splits are present
    if not train_data:
        raise ValueError("Training dataset is empty or not found.")
    if not validation_data:
        raise ValueError("Validation dataset is empty or not found.")   
    if not test_data:
        raise ValueError("Test dataset is empty or not found.")

    print(f"Dataset loaded successfully:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(validation_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Create dataset objects
    train_dataset = FlickrDataset(train_data)
    val_dataset = FlickrDataset(validation_data)
    test_dataset = FlickrDataset(test_data)
    
    # Start training
    trained_model, eval_results = train(model, train_dataset, val_dataset, wandb.config, device)
    
    print("Training completed!")
    print(f"Final validation loss: {eval_results['avg_loss']:.4f}")
    
    # Final evaluation on test set
    print("Running final evaluation on test set...")
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=collate_fn)
    token_converter = TokenConverter("roberta-base")
    final_metrics = evaluate_model(trained_model, test_loader, device, token_converter, save_dir="final_eval_results")
    
    print(f"Test set loss: {final_metrics['avg_loss']:.4f}")
    wandb.log({"final_test_loss": final_metrics['avg_loss']})
    
    wandb.finish()


if __name__ == "__main__":
    main()

### PSEUDOCODE FOR THIS IMPLEMENTATION
# 1. Load the necessary libraries and modules.
# 2. Define the `train` function that will handle the training logic.
#    2.1 Train recieves a decoder model as an argument.
#    2.2 Train recieves a dataset as an argument.
#    2.3 Train for one epoch and calculate cross-entropy loss of output tokens and target tokens
#    2.4 Backpropagate the loss and update the model parameters.
#    2.5 Save the trained model and embeddings.
#    2.6 Return the trained model.
#    2.7 Return the embeddings.
#    2.8 Return the loss.
#    2.9 Run an eval on test dataset and return the results.
#.        2.9.1 Eval function should show the image, the caption, and the predicted caption for the top 3 and bottom 3 predictions and 3 random predictions.
#    2.10 Return the eval results.
# 3. Define the `main` function to set up the training environment.
#    3.1 Initialize Weights & Biases for experiment tracking.
#    3.2 Define the configuration for the decoder model.
#.   3.3 Load the dataset from a pickle file.
#    3.4 Check if the dataset is loaded correctly and contains the necessary splits.
#    3.5 Create an instance of the `DecoderFull` model with the specified
#.   3.6 Change dataset to a torch object, use a dataloader for batching