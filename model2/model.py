from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Model, CLIPVisionModel, GPT2LMHeadModel
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPVisionTransformer
from transformers.utils import can_return_tuple
import torch
from torch import nn
from transformers.models.clip.modeling_clip import BaseModelOutput
from typing import Optional

# Import FlickrDataset from the model package (different from model2)
from model.dataset import FlickrDataset, qwen_collate_fn

def extract_CLIP_image_embeddings(save_path="data/clip_image_embeddings"):

    full_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    
    # Extract the transformer sub-module and get its state_dict directly
    embeddings_state_dict = full_model.vision_model.state_dict()
    
    print("🎯 Extracted embeddings state_dict:")
    for k, v in embeddings_state_dict.items():
        print(f"  {k}: {v.shape}")
    
    config = full_model.config

    del full_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return embeddings_state_dict, config

class ClipModule(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__(config)
        # Replace post_layernorm with identity instead of deleting it
        self.post_layernorm = nn.Identity()

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        """Load from pretrained CLIP model and adapt to our custom architecture"""
        # Load the full pretrained model
        full_model = CLIPVisionModel.from_pretrained(model_name_or_path)
        
        # Create our custom instance
        custom_model = cls(full_model.config)
        
        # Transfer weights from pretrained model
        custom_model.load_state_dict(full_model.vision_model.state_dict(), strict=False)
        
        # Clean up
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return custom_model

    @can_return_tuple
    def forward(self, pixel_values: Optional[torch.FloatTensor] = None, interpolate_pos_encoding: bool = False, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return encoder_outputs



class QwenModelv2(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B-Base"):
        super().__init__()
        self.qwen = Qwen3Model.from_pretrained(model_name)
        
        self.image_encoder = ClipModule.from_pretrained("openai/clip-vit-large-patch14")
        
        # Add projection layer to match dimensions
        self.image_projection = nn.Linear(1024, self.qwen.config.hidden_size)  # CLIP: 1024 -> Qwen: varies
        
        # Add language modeling head to generate vocabulary logits
        self.lm_head = nn.Linear(self.qwen.config.hidden_size, self.qwen.config.vocab_size, bias=False)
        

    def forward(self, image_data, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # Process image 
        encoded_image_embeddings = self.image_encoder(image_data).last_hidden_state  # [batch, 257, 1024]
        image_tokens = self.image_projection(encoded_image_embeddings)  # [batch, 257, hidden_size]
        
        # Get text embeddings from Qwen
        text_embeddings = self.qwen.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]
        
        # Concatenate image and text embeddings
        combined_embeddings = torch.cat([image_tokens, text_embeddings], dim=1)
        
        # Create combined attention mask
        # Create image attention mask efficiently 
        image_attention = torch.ones(
            batch_size, image_tokens.shape[1], 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        # Forward through Qwen with combined inputs
        outputs = self.qwen(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask
        )
        
        # Apply language modeling head to get vocabulary logits
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        return logits
