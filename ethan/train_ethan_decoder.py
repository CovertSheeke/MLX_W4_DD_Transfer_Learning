import torch
import wandb
from ethan_decoder_model import DecoderFull
from dotenv import load_dotenv
import os
import pickle

load_dotenv(override=True)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "")


def train(decoder):
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

    # # Print the model architecture
    # print(model)

    # #load pre-embedded dataset here
    # data_path = os.path.join('..', 'flickr30k_processed_splits_test.pkl')
    data_path = 'flickr30k_processed_splits_test.pkl'

    with open(data_path, 'rb') as f:
        raw_dataset = pickle.load(f)

    # Check if the dataset is loaded correctly
    if not raw_dataset:
        raise ValueError("Dataset is empty or not loaded correctly.")
    
    # Assuming the dataset is a dictionary with 'train', 'validation', and 'test' splits
    train_dataset = raw_dataset.get('train', [])
    validation_dataset = raw_dataset.get('validation', [])
    test_dataset = raw_dataset.get('test', [])

    # Check if the splits are present
    if not train_dataset:
        raise ValueError("Training dataset is empty or not found.")
    if not validation_dataset:
        raise ValueError("Validation dataset is empty or not found.")   
    if not test_dataset:
        raise ValueError("Test dataset is empty or not found.")

    # if not os.path.exists(data_path):
    #     raise FileNotFoundError(f"Dataset file not found: {data_path}")
    # Load the dataset
    # with open(data_path, 'rb') as f:
    #     dataset = torch.load(f)


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

### IDEAL PSEUDOCODE
# 1. Load the un-embedded dataset from a pickle file.
# 2. Initialize the Weights & Biases project.
# 3. Create the decoder model with the specified configuration.
# 4. Load the pre-trained text and image encoders.
# 5. Train the decoder model using the loaded dataset.
    # 5.1 For each batch in the dataset:
#         - Extract the image and caption pairs.
#         - Pass the images through the image encoder to get image embeddings.
#         - Pass the captions through the text encoder to get text embeddings.
#         - Add start tokens to the captions, and end tokens to the targets.
#         - Pad the captions and targets to the maximum sequence length.
#         - Pass the image embeddings and padded captions through the decoder.
#         - Compute the loss using the decoder's output and the targets.
#         - Backpropagate the loss and update the model parameters.
# 6. Save the trained model and the embeddings.