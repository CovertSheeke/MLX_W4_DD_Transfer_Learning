"""
Modular Image-Text Dataset Processing Pipeline

This module provides a flexible framework for processing image-text datasets
with different vision and text encoders, automatic dimension alignment,
and configurable padding strategies.
"""

from datasets import load_dataset, DatasetDict, Dataset
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel
from PIL import Image
from tqdm import tqdm
import numpy as np


class EmbeddingProcessor:
    """Handles embedding generation for images and text using specified models."""
    
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "roberta-base",
        device: str = "cpu"
    ):
        self.device = device
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        
        # Load vision encoder and processor
        self.vision_encoder = CLIPModel.from_pretrained(vision_model_name).to(device)
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model_name)
        
        # Load text encoder and tokenizer
        self.text_encoder = RobertaModel.from_pretrained(text_model_name).to(device)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
        
        # Get embedding dimensions (handle different model architectures)
        if hasattr(self.vision_encoder.config, 'hidden_size'):
            self.vision_dim = self.vision_encoder.config.hidden_size
        elif hasattr(self.vision_encoder.config, 'vision_config'):
            self.vision_dim = self.vision_encoder.config.vision_config.hidden_size
        else:
            # Fallback: get dimension from a test forward pass
            test_input = torch.zeros(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                test_output = self.vision_encoder.vision_model(pixel_values=test_input)
                self.vision_dim = test_output.last_hidden_state.shape[-1]
        
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Create projection layer if dimensions don't match
        self.projection = None
        if self.vision_dim != self.text_dim:
            self.projection = nn.Linear(self.vision_dim, self.text_dim).to(device)
            print(f"Created projection layer: {self.vision_dim} -> {self.text_dim}")
        
        print(f"Vision encoder: {vision_model_name} (dim: {self.vision_dim})")
        print(f"Text encoder: {text_model_name} (dim: {self.text_dim})")
    
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """Generate embeddings for a single image."""
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vision_encoder.vision_model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0)  # [num_patches, hidden_dim]
            
            # Apply projection if needed
            if self.projection is not None:
                embeddings = self.projection(embeddings)
        
        return embeddings.cpu()
    
    def embed_text(self, text: str, return_targets: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate embeddings for text with optional autoregressive targets."""
        inputs = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        input_ids = inputs["input_ids"].squeeze(0)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
        
        if return_targets:
            # Prepare for autoregressive training
            text_embeds = embeddings[:-1].cpu()  # Remove last token
            target_ids = input_ids[1:].cpu()     # Remove first token
            return text_embeds, target_ids
        
        return embeddings.cpu()
    
    def get_padding_embedding(self) -> torch.Tensor:
        """Get the embedding for the padding token."""
        pad_token = self.text_tokenizer.pad_token
        if pad_token is None:
            raise ValueError("Text tokenizer does not have a padding token")
        
        pad_inputs = self.text_tokenizer.encode(pad_token, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            pad_outputs = self.text_encoder(pad_inputs)
            pad_embedding = pad_outputs.last_hidden_state.squeeze(0)[0]  # [hidden_dim]
        
        return pad_embedding.cpu()


class DatasetProcessor:
    """Handles dataset loading, processing, and saving."""
    
    def __init__(
        self,
        embedding_processor: EmbeddingProcessor,
        image_column: str = "image",
        caption_column: str = "caption",
        max_samples: Optional[int] = None
    ):
        self.embedding_processor = embedding_processor
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_samples = max_samples
        self.max_text_length = 0
        self.padding_embedding = None
    
    def process_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of image-caption pairs."""
        images = []
        captions = []
        image_embeds = []
        text_embeds = []
        
        # Handle nested caption structure (like Flickr30k)
        batch_images = batch[self.image_column]
        batch_captions = batch[self.caption_column]
        
        for img_idx, (image, caps) in enumerate(tqdm(
            zip(batch_images, batch_captions), 
            total=len(batch_images),
            desc="Processing batch",
            leave=False
        )):
            # Handle both single captions and lists of captions
            if isinstance(caps, str):
                caps = [caps]
            elif not isinstance(caps, list):
                caps = list(caps)
            
            # Generate image embedding once per image
            img_embed = self.embedding_processor.embed_image(image)
            
            # Process each caption for this image
            for caption in caps:
                images.append(image)
                captions.append(caption)
                image_embeds.append(img_embed)
                
                # Generate text embedding (without autoregressive targets for now)
                text_embed = self.embedding_processor.embed_text(caption, return_targets=False)
                text_embeds.append(text_embed)
        
        return {
            self.image_column: images,
            self.caption_column: captions,
            "image_embed": image_embeds,
            "text_embed": text_embeds
        }
    
    def calculate_max_length(self, dataset: Dataset) -> int:
        """Calculate maximum text sequence length in the dataset."""
        max_length = 0
        for item in tqdm(dataset, desc="Calculating max sequence length"):
            text_embed = item["text_embed"]
            if isinstance(text_embed, list):
                max_length = max(max_length, len(text_embed))
            else:
                max_length = max(max_length, text_embed.shape[0])
        return max_length
    
    def pad_text_embeddings(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Pad text embeddings to max length."""
        if self.padding_embedding is None:
            self.padding_embedding = self.embedding_processor.get_padding_embedding()
        
        text_embed = example["text_embed"]
        
        # Convert to list if it's a tensor
        if isinstance(text_embed, torch.Tensor):
            text_embed = text_embed.tolist()
        
        current_length = len(text_embed)
        if current_length < self.max_text_length:
            padding_needed = self.max_text_length - current_length
            pad_embed_list = self.padding_embedding.tolist()
            text_embed.extend([pad_embed_list] * padding_needed)
        
        example["text_embed"] = text_embed
        return example
    
    def process_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        split_name: Optional[str] = None
    ) -> Dataset:
        """Process a complete dataset or a specific split."""
        
        # Handle DatasetDict vs Dataset
        if isinstance(dataset, DatasetDict):
            if split_name is None:
                raise ValueError("Must specify split_name when processing DatasetDict")
            working_dataset = dataset[split_name]
        else:
            working_dataset = dataset
        
        # Limit samples if specified
        if self.max_samples is not None:
            working_dataset = working_dataset.select(range(min(self.max_samples, len(working_dataset))))
        
        # Process embeddings
        print(f"Processing {len(working_dataset)} samples...")
        processed = working_dataset.map(
            self.process_batch,
            batched=True,
            remove_columns=working_dataset.column_names,
            desc="Generating embeddings"
        )
        
        # Calculate max length and apply padding
        print("Calculating maximum text sequence length...")
        self.max_text_length = self.calculate_max_length(processed)
        print(f"Maximum text sequence length: {self.max_text_length}")
        
        print("Applying padding to text embeddings...")
        processed = processed.map(self.pad_text_embeddings, desc="Padding sequences")
        
        return processed


def create_train_val_test_splits(
    dataset: Dataset,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    shuffle: bool = True,
    seed: int = 42
) -> DatasetDict:
    """Split a dataset into train/validation/test splits."""
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_ds = dataset.select(range(n_train))
    val_ds = dataset.select(range(n_train, n_train + n_val))
    test_ds = dataset.select(range(n_train + n_val, n))
    
    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })


def convert_to_tensors(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dataset items back to tensors for training."""
    result = item.copy()
    
    # Convert embeddings to tensors
    if isinstance(item["image_embed"], list):
        result["image_embed"] = torch.tensor(item["image_embed"])
    if isinstance(item["text_embed"], list):
        result["text_embed"] = torch.tensor(item["text_embed"])
    
    return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for DataLoader."""
    # Convert all items to tensors first
    batch = [convert_to_tensors(item) for item in batch]
    
    # Stack tensors
    image_embeds = torch.stack([item["image_embed"] for item in batch])
    text_embeds = torch.stack([item["text_embed"] for item in batch])
    
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "image_embeds": image_embeds,  # [batch_size, num_patches, hidden_dim]
        "text_embeds": text_embeds     # [batch_size, max_length, hidden_dim]
    }


def main():
    """Main processing pipeline for Flickr30k dataset."""
    
    # Configuration
    CONFIG = {
        "dataset_name": "nlphuji/flickr30k",
        "vision_model": "openai/clip-vit-base-patch32",
        "text_model": "roberta-base",
        "image_column": "image",
        "caption_column": "caption",
        "max_samples": None,  # Set to None for full dataset
        "output_file": "flickr30k_processed_splits.pkl",
        "device": "cpu"
    }
    
    print("=== Image-Text Dataset Processing Pipeline ===")
    print(f"Dataset: {CONFIG['dataset_name']}")
    print(f"Vision Model: {CONFIG['vision_model']}")
    print(f"Text Model: {CONFIG['text_model']}")
    print(f"Max Samples: {CONFIG['max_samples']}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(CONFIG["dataset_name"])
    print(f"Available splits: {list(dataset.keys())}")
    
    # Initialize processors
    embedding_processor = EmbeddingProcessor(
        vision_model_name=CONFIG["vision_model"],
        text_model_name=CONFIG["text_model"],
        device=CONFIG["device"]
    )
    
    dataset_processor = DatasetProcessor(
        embedding_processor=embedding_processor,
        image_column=CONFIG["image_column"],
        caption_column=CONFIG["caption_column"],
        max_samples=CONFIG["max_samples"]
    )
    
    # Process dataset (assume single 'test' split for Flickr30k)
    if "test" in dataset and len(dataset) == 1:
        print("Processing single 'test' split...")
        processed = dataset_processor.process_dataset(dataset, "test")
        
        # Create train/val/test splits
        print("Creating train/validation/test splits...")
        split_dataset = create_train_val_test_splits(processed)
        
    else:
        # Handle datasets that already have splits
        print("Processing existing splits...")
        split_dataset = DatasetDict()
        for split_name in dataset.keys():
            print(f"Processing {split_name} split...")
            split_dataset[split_name] = dataset_processor.process_dataset(dataset, split_name)
    
    # Print split sizes
    print("\nDataset split sizes:")
    for split_name, split_data in split_dataset.items():
        print(f"  {split_name}: {len(split_data):,} samples")
    
    # Verify embedding shapes
    print("\nVerifying embedding shapes...")
    for split_name in ["train", "validation", "test"]:
        if split_name in split_dataset:
            print(f"\n{split_name.capitalize()} split:")
            for i in range(min(3, len(split_dataset[split_name]))):
                item = split_dataset[split_name][i]
                img_embed = torch.tensor(item["image_embed"])
                text_embed = torch.tensor(item["text_embed"])
                
                print(f"  Item {i}: image_embed={img_embed.shape}, text_embed={text_embed.shape}")
                print(f"  Item {i}: caption='{item['caption'][:50]}...'")
    
    # Save processed dataset
    output_path = Path(CONFIG["output_file"])
    print(f"\nSaving processed dataset to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(split_dataset, f)
    
    print("✅ Processing complete!")


if __name__ == "__main__":
    main()
