#!/usr/bin/env python3
"""
Inference script for Vision-Language Model
Generates captions for images using trained Qwen + CLIP model
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from pathlib import Path
import requests
from io import BytesIO
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor
from typing import Optional, List
import json

from .model import QwenModel


class VisionLanguageInference:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize inference pipeline
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"🔧 Using device: {self.device}")
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Setup image processor (using CLIP's processor for consistency)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        print("✅ Inference pipeline ready!")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model and tokenizer from checkpoint"""
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'Qwen/Qwen3-0.6B-Base')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add special tokens if they were used during training
        special_tokens = {'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize model
        self.model = QwenModel(model_name)
        
        # Resize embeddings to match tokenizer (in case special tokens were added)
        self.model.qwen.resize_token_embeddings(len(self.tokenizer))
        
        # Update lm_head if vocabulary size changed
        vocab_size = len(self.tokenizer)
        if hasattr(self.model, 'lm_head') and self.model.lm_head.out_features != vocab_size:
            from torch import nn
            self.model.lm_head = nn.Linear(self.model.qwen.config.hidden_size, vocab_size, bias=False)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('global_step', 'unknown')}")
        print(f"📝 Tokenizer vocab size: {len(self.tokenizer)}")
        
    def load_image(self, image_source: str) -> Image.Image:
        """
        Load image from file path or URL
        
        Args:
            image_source: Path to local image file or URL
            
        Returns:
            PIL Image object
        """
        try:
            if image_source.startswith(('http://', 'https://')):
                # Load from URL
                print(f"🌐 Loading image from URL: {image_source}")
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # Load from local file
                print(f"📁 Loading image from file: {image_source}")
                image = Image.open(image_source)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_source}: {e}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        # Use CLIP's image processor
        inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def generate_caption(
        self, 
        image_source: str,
        prompt: str = "Describe this image:",
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Generate caption for an image
        
        Args:
            image_source: Path to image file or URL
            prompt: Text prompt to guide caption generation
            max_length: Maximum length of generated caption
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling (False = greedy)
            
        Returns:
            Generated caption text
        """
        # Load and preprocess image
        image = self.load_image(image_source)
        image_tensor = self.preprocess_image(image)
        
        # Prepare text input
        text_input = f"<|im_start|>{prompt}<|im_end|>"
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        print(f"🎯 Generating caption with prompt: '{prompt}'")
        
        with torch.no_grad():
            # Get initial hidden states from the model
            outputs = self.model(
                image_data=image_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use autoregressive generation
            generated_sequence = input_ids.clone()
            
            for step in range(max_length):
                # Forward pass
                current_outputs = self.model(
                    image_data=image_tensor,
                    input_ids=generated_sequence,
                    attention_mask=torch.ones_like(generated_sequence)
                )
                
                # Get next token logits
                next_token_logits = current_outputs[:, -1, :]
                
                # Apply repetition penalty
                if step > 0:
                    for token_id in generated_sequence[0].unique():
                        if token_id in generated_sequence[0]:
                            next_token_logits[0, token_id] /= repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to sequence
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
            # Decode generated text (skip the original prompt)
            original_length = input_ids.shape[1]
            new_tokens = generated_sequence[:, original_length:]
            generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            
        return generated_text.strip()
    
    def batch_generate_captions(
        self,
        image_sources: List[str],
        prompt: str = "Describe this image:",
        **generation_kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images
        
        Args:
            image_sources: List of image paths or URLs
            prompt: Text prompt for all images
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated captions
        """
        captions = []
        
        for i, image_source in enumerate(image_sources):
            print(f"\n📸 Processing image {i+1}/{len(image_sources)}: {Path(image_source).name}")
            try:
                caption = self.generate_caption(image_source, prompt, **generation_kwargs)
                captions.append(caption)
                print(f"💬 Caption: {caption}")
            except Exception as e:
                print(f"❌ Error processing {image_source}: {e}")
                captions.append(f"Error: {str(e)}")
        
        return captions


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using trained vision-language model')
    
    # Required arguments
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint file')
    parser.add_argument('image', type=str, help='Path to image file or URL (or directory for batch processing)')
    
    # Optional arguments
    parser.add_argument('--prompt', type=str, default='Describe this image:', 
                        help='Text prompt to guide caption generation')
    parser.add_argument('--max_length', type=int, default=100, 
                        help='Maximum length of generated caption')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                        help='Penalty for repeated tokens')
    parser.add_argument('--no_sampling', action='store_true',
                        help='Use greedy decoding instead of sampling')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in directory (if image is a directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = VisionLanguageInference(args.checkpoint, args.device)
    
    # Generation parameters
    generation_kwargs = {
        'prompt': args.prompt,
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
        'do_sample': not args.no_sampling
    }
    
    # Process images
    image_path = Path(args.image)
    results = []
    
    if args.batch and image_path.is_dir():
        # Batch processing
        print(f"🗂️ Batch processing images in {image_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in image_path.glob('*') if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("❌ No images found in directory!")
            return
        
        print(f"📸 Found {len(image_files)} images")
        captions = inference.batch_generate_captions(
            [str(f) for f in image_files], 
            **generation_kwargs
        )
        
        for image_file, caption in zip(image_files, captions):
            results.append({
                'image': str(image_file),
                'caption': caption,
                'prompt': args.prompt
            })
    
    else:
        # Single image processing
        try:
            caption = inference.generate_caption(str(image_path), **generation_kwargs)
            print(f"\n🎯 Generated Caption:")
            print(f"📸 Image: {image_path}")
            print(f"💬 Caption: {caption}")
            
            results.append({
                'image': str(image_path),
                'caption': caption,
                'prompt': args.prompt
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")


if __name__ == '__main__':
    main() 