import torch
import wandb
from ethan_decoder_model import DecoderFull
from dotenv import load_dotenv
import os

load_dotenv(override=True)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "")


def train(decoder, img_encoder, text_encoder):
    # Placeholder for training logic
    

    pass


def main():
    # Initialize Weights & Biases
    wandb.init(project="ethan_decoder_training", entity="",
               config = {
        "seq_len": 128,
        "dim_attn_in": 512,  # Output dimension of the attention mechanism
        "num_decoder_blocks": 6,
        "dim_ffn": 2048,
        "num_msAttnHeads": 8,
        "dim_V": 64,  # Example dimension for value vectors
        "dim_KQ": 64,
        "dim_logits": 1000,  # Example output dimension
        "dropout_rate": 0.1
    })

    # Define the configuration for the model
    
    ## load text encoder and image encoder models here if needed
    # Create the model
    model = DecoderFull(wandb.config)

    # Print the model architecture
    print(model)

if __name__ == "__main__":
    main()