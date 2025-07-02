from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Model, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
import torch
from torch import nn


def extract_CLIP_image_embeddings(save_path="data/clip_image_embeddings"):

    full_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    embeddings_state_dict = {
        k: v for k, v in full_model.state_dict().items() 
        if k.startswith("embeddings.patch_embedding")
    }
    config = full_model.config

    del full_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return embeddings_state_dict, config

class QwenModel(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B-Base"):
        super().__init__()
        self.qwen = Qwen3Model.from_pretrained(model_name)
        embeddings_state_dict, config = extract_CLIP_image_embeddings() 
        self.image_embed = CLIPVisionEmbeddings(config)
        self.image_embed.load_state_dict(embeddings_state_dict)
        
        # Add projection layer to match dimensions
        self.image_projection = nn.Linear(1024, self.qwen.config.hidden_size)  # CLIP: 1024 -> Qwen: varies
        
        # Add language modeling head to generate vocabulary logits
        self.lm_head = nn.Linear(self.qwen.config.hidden_size, self.qwen.config.vocab_size, bias=False)
        
        # Make models trainable
        for param in self.qwen.parameters():
            param.requires_grad = True
        for param in self.image_embed.parameters():
            param.requires_grad = True

    def forward(self, image_data, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # Process image embeddings
        image_embeddings = self.image_embed(image_data)  # [batch, 257, 1024]
        image_embeddings = self.image_projection(image_embeddings)  # [batch, 257, hidden_size]
        
        # Get text embeddings from Qwen
        text_embeddings = self.qwen.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]
        
        # Concatenate image and text embeddings
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # Create combined attention mask
        image_attention = torch.ones(batch_size, image_embeddings.shape[1], device=attention_mask.device)
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
