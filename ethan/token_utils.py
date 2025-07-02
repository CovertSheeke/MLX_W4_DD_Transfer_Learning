"""
Token Utility Functions

This module provides utilities for converting tokens back to text using different approaches:
1. Direct tokenizer-based conversion (requires loading the full model)
2. Efficient dictionary-based conversion (precomputed for runtime efficiency)
"""

import pickle
import torch
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, RobertaTokenizer
from pathlib import Path


class TokenConverter:
    """
    A utility class for converting tokens back to text using different strategies.
    """
    
    def __init__(self, tokenizer_name: str = "roberta-base"):
        """
        Initialize the token converter.
        
        Args:
            tokenizer_name: Name of the tokenizer to use (e.g., "roberta-base")
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.token_to_text_dict = None
        self._special_tokens = None
    
    def load_tokenizer(self):
        """Load the tokenizer for direct conversion."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            print(f"Loaded tokenizer: {self.tokenizer_name}")
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token information."""
        if self._special_tokens is None:
            self.load_tokenizer()
            self._special_tokens = {
                'pad_token': self.tokenizer.pad_token,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token': self.tokenizer.eos_token,
                'eos_token_id': self.tokenizer.eos_token_id,
                'bos_token': getattr(self.tokenizer, 'bos_token', None),
                'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
                'unk_token': self.tokenizer.unk_token,
                'unk_token_id': self.tokenizer.unk_token_id,
                'cls_token': getattr(self.tokenizer, 'cls_token', None),
                'cls_token_id': getattr(self.tokenizer, 'cls_token_id', None),
                'sep_token': getattr(self.tokenizer, 'sep_token', None),
                'sep_token_id': getattr(self.tokenizer, 'sep_token_id', None),
                'mask_token': getattr(self.tokenizer, 'mask_token', None),
                'mask_token_id': getattr(self.tokenizer, 'mask_token_id', None),
            }
        return self._special_tokens
    
    def tokens_to_text_direct(
        self, 
        tokens: Union[List[int], torch.Tensor], 
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Convert tokens to text using the tokenizer directly.
        
        Args:
            tokens: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens in decoding
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces
            
        Returns:
            Decoded text string
        """
        self.load_tokenizer()
        
        # Convert tensor to list if needed
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # Remove padding tokens if present
        if skip_special_tokens and self.tokenizer.pad_token_id is not None:
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]
        
        return self.tokenizer.decode(
            tokens, 
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def create_token_dictionary(self, save_path: Optional[str] = None) -> Dict[int, str]:
        """
        Create a dictionary mapping token IDs to their text representations.
        This is more efficient for runtime conversion when you don't need the full tokenizer.
        
        Args:
            save_path: Optional path to save the dictionary as a pickle file
            
        Returns:
            Dictionary mapping token IDs to text
        """
        self.load_tokenizer()
        
        print("Creating token ID to text dictionary...")
        vocab_size = self.tokenizer.vocab_size
        token_dict = {}
        
        # Get all tokens in the vocabulary
        for token_id in range(vocab_size):
            try:
                # Decode individual token
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                token_dict[token_id] = token_text
            except Exception as e:
                print(f"Warning: Could not decode token ID {token_id}: {e}")
                token_dict[token_id] = f"<UNK_{token_id}>"
        
        # Add any additional special tokens
        special_tokens = self.get_special_tokens()
        for token_name, token_id in special_tokens.items():
            if token_id is not None and token_id not in token_dict:
                token_text = special_tokens.get(token_name.replace('_id', ''), f'<{token_name}>')
                token_dict[token_id] = token_text
        
        self.token_to_text_dict = token_dict
        
        if save_path:
            save_path = Path(save_path)
            print(f"Saving token dictionary to {save_path}...")
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'tokenizer_name': self.tokenizer_name,
                    'token_dict': token_dict,
                    'special_tokens': special_tokens,
                    'vocab_size': vocab_size
                }, f)
            print(f"Token dictionary saved with {len(token_dict)} entries")
        
        return token_dict
    
    def load_token_dictionary(self, dict_path: str):
        """
        Load a precomputed token dictionary from file.
        
        Args:
            dict_path: Path to the saved token dictionary pickle file
        """
        print(f"Loading token dictionary from {dict_path}...")
        with open(dict_path, 'rb') as f:
            data = pickle.load(f)
        
        self.token_to_text_dict = data['token_dict']
        self._special_tokens = data['special_tokens']
        
        print(f"Loaded token dictionary with {len(self.token_to_text_dict)} entries")
        print(f"Original tokenizer: {data['tokenizer_name']}")
    
    def tokens_to_text_dict(
        self, 
        tokens: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        join_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Convert tokens to text using the precomputed dictionary (faster for runtime).
        
        Args:
            tokens: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens
            join_tokens: Whether to join tokens into a single string
            
        Returns:
            Decoded text string or list of token texts
        """
        if self.token_to_text_dict is None:
            raise ValueError("Token dictionary not loaded. Call create_token_dictionary() or load_token_dictionary() first.")
        
        # Convert tensor to list if needed
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # Convert tokens to text
        token_texts = []
        special_token_ids = set()
        
        if skip_special_tokens and self._special_tokens:
            special_token_ids = {v for k, v in self._special_tokens.items() 
                               if k.endswith('_id') and v is not None}
        
        for token_id in tokens:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            token_text = self.token_to_text_dict.get(token_id, f"<UNK_{token_id}>")
            token_texts.append(token_text)
        
        if join_tokens:
            # Simple join - may need more sophisticated handling for subword tokens
            return ' '.join(token_texts).replace(' ##', '').strip()
        else:
            return token_texts
    
    def batch_tokens_to_text(
        self, 
        batch_tokens: Union[List[List[int]], torch.Tensor],
        method: str = "direct",
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Convert a batch of token sequences to text.
        
        Args:
            batch_tokens: Batch of token sequences
            method: "direct" to use tokenizer, "dict" to use dictionary
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        if isinstance(batch_tokens, torch.Tensor):
            batch_tokens = batch_tokens.tolist()
        
        if method == "direct":
            return [self.tokens_to_text_direct(tokens, skip_special_tokens) 
                   for tokens in batch_tokens]
        elif method == "dict":
            return [self.tokens_to_text_dict(tokens, skip_special_tokens) 
                   for tokens in batch_tokens]
        else:
            raise ValueError("Method must be 'direct' or 'dict'")


def create_roberta_token_dict(save_path: str = "roberta_token_dict.pkl"):
    """
    Convenience function to create and save a RoBERTa token dictionary.
    
    Args:
        save_path: Path to save the dictionary
    """
    converter = TokenConverter("roberta-base")
    converter.create_token_dictionary(save_path)
    return converter


def demo_token_conversion():
    """Demonstrate token conversion functionality."""
    print("=== Token Conversion Demo ===\n")
    
    # Create converter
    converter = TokenConverter("roberta-base")
    
    # Example tokens (this would typically come from your dataset)
    example_text = "A dog is running in the park."
    
    # First, let's tokenize some text to get tokens
    converter.load_tokenizer()
    tokens = converter.tokenizer.encode(example_text, add_special_tokens=True)
    print(f"Original text: '{example_text}'")
    print(f"Tokens: {tokens}")
    
    # Method 1: Direct conversion
    print("\n--- Method 1: Direct Tokenizer Conversion ---")
    decoded_direct = converter.tokens_to_text_direct(tokens)
    print(f"Decoded (direct): '{decoded_direct}'")
    
    # Method 2: Dictionary-based conversion
    print("\n--- Method 2: Dictionary-based Conversion ---")
    token_dict = converter.create_token_dictionary()
    decoded_dict = converter.tokens_to_text_dict(tokens)
    print(f"Decoded (dict): '{decoded_dict}'")
    
    # Show special tokens
    print("\n--- Special Tokens ---")
    special_tokens = converter.get_special_tokens()
    for name, value in special_tokens.items():
        if value is not None:
            print(f"  {name}: {value}")
    
    # Batch conversion example
    print("\n--- Batch Conversion Example ---")
    batch_tokens = [tokens, tokens[:5], tokens[2:8]]  # Different length sequences
    batch_decoded = converter.batch_tokens_to_text(batch_tokens, method="direct")
    for i, text in enumerate(batch_decoded):
        print(f"  Batch item {i}: '{text}'")


if __name__ == "__main__":
    demo_token_conversion()
