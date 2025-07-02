from datasets import load_dataset
import pickle
from datasets import DatasetDict

from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel
from PIL import Image
import torch
import requests
from tqdm import tqdm
# Load encoders
vision_enc = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
text_enc = RobertaModel.from_pretrained("roberta-base")

# Load processors
vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


ds = load_dataset("nlphuji/flickr30k")

with open("flickr30k_dataset.pkl", "wb") as f:
    pickle.dump(ds, f)

# To load the dataset later, you can use:
with open("flickr30k_dataset.pkl", "rb") as f:
    ds = pickle.load(f)

# Open the dataset from pickle
with open("flickr30k_dataset.pkl", "rb") as f:
    ds = pickle.load(f)

def embed_image(image: Image.Image, vision_enc, vision_processor) -> torch.Tensor:
    """
    Given a PIL image, returns image token embeddings from the vision encoder.
    Output shape: [num_image_tokens, hidden_dim] (squeezed batch dim)
    """
    image_inputs = vision_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_outputs = vision_enc.vision_model(**image_inputs)
    image_embeds = image_outputs.last_hidden_state.squeeze(0)  # [num_patches+1, hidden_dim]
    return image_embeds

def embed_text(text: str, text_enc, text_tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a string of text, returns:
    - token embeddings from the text encoder (without final token)
    - shifted input_ids for training targets

    Returns:
    - text_embeds: [text_len - 1, hidden_dim]
    - target_ids:  [text_len - 1] (int tokens)
    """
    text_inputs = text_tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = text_inputs["input_ids"].squeeze(0)  # [text_len]
    with torch.no_grad():
        text_outputs = text_enc(**text_inputs)
    text_embeds = text_outputs.last_hidden_state.squeeze(0)  # [text_len, hidden_dim]

    # Prepare for training (autoregressive shift)
    text_embeds = text_embeds[:-1]  # Remove last token
    target_ids = input_ids[1:]      # Remove first token

    return text_embeds, target_ids

# Placeholder for preprocessing logic
def all_pairs(batch):
    # batch["image"] is a list of images, batch["caption"] is a list of lists of captions, batch["img_id"] is a list of image ids
    images = []
    captions = []
    img_ids = []
    sentids = []
    img_embeds = []
    text_embeds = []

    for image, caps, img_id, sids in tqdm(
        zip(batch["image"], batch["caption"], batch["img_id"], batch["sentids"]),
        total=len(batch["image"]),
        desc="Processing batch"
    ):
        images.extend([image] * len(caps))
        captions.extend(caps)
        img_ids.extend([img_id] * len(caps))
        sentids.extend(sids)
        img_embed = embed_image(image, vision_enc, vision_processor)
        # Keep as tensor for each caption
        for cap in caps:
            img_embeds.append(img_embed)  # Store as tensor
            text_embed = embed_text(cap, text_enc, text_tokenizer)[0]  # Get only the embeddings
            text_embeds.append(text_embed)  # Store as tensor

    return {"image": images, "caption": captions, "img_id": img_ids, "sentids": sentids, "image_embed": img_embeds, "text_embed": text_embeds}

# Check which splits are available in the dataset
print("Available splits:", ds.keys())

# If only 'test' split is available, split it into train/validation/test
if "test" in ds and len(ds) == 1:
    test_dataset = ds["test"]
    n = len(test_dataset)
    n_train = int(n * 0.75)
    n_val = int(n * 0.10)
    n_test = n - n_train - n_val

    # Shuffle before splitting
    test_dataset = test_dataset.shuffle(seed=42)
    train_ds = test_dataset.select(range(n_train))
    val_ds = test_dataset.select(range(n_train, n_train + n_val))
    test_ds = test_dataset.select(range(n_train + n_val, n))

    ds = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
    # Shuffle and preprocess before splitting
    if "test" in ds:
        print
        # Shuffle and select the first 1000 rows of the 'test' split
        shuffled = ds["test"].shuffle(seed=42)
        small_test = shuffled.select(range(min(50, len(shuffled))))
        # Preprocess
        processed = small_test.map(
            all_pairs,
            batched=True,
            remove_columns=small_test.column_names,
        )
        processed = processed.shuffle(seed=42)
        
        # Calculate max sequence length and pad all text embeddings
        print("\nCalculating maximum sequence length...")
        max_length = 0
        for item in processed:
            text_embed_len = len(item["text_embed"])
            max_length = max(max_length, text_embed_len)
        
        print(f"Maximum text sequence length: {max_length}")
        
        # Get padding embedding (embedding for the padding token)
        pad_token_id = text_tokenizer.pad_token_id
        pad_inputs = text_tokenizer.encode(text_tokenizer.pad_token, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            pad_outputs = text_enc(pad_inputs)
        pad_embedding = pad_outputs.last_hidden_state.squeeze(0)[0]  # [hidden_dim]
        pad_embedding_np = pad_embedding.cpu().numpy()
        
        # Function to pad text embeddings
        def pad_text_embeddings(example):
            text_embed = example["text_embed"]
            current_length = len(text_embed)
            if current_length < max_length:
                # Add padding embeddings
                padding_needed = max_length - current_length
                padded_embed = text_embed + [pad_embedding_np.tolist()] * padding_needed
                example["text_embed"] = padded_embed
            
            return example
        
        # Apply padding to the processed dataset
        print("Applying padding to text embeddings...")
        processed = processed.map(pad_text_embeddings)
        print(f"All text embeddings now have length: {max_length}")
        
        # Now split into train/validation/test
        n = len(processed)
        n_train = int(n * 0.75)
        n_val = int(n * 0.10)
        n_test = n - n_train - n_val

        train_ds = processed.select(range(n_train))
        val_ds = processed.select(range(n_train, n_train + n_val))
        test_ds = processed.select(range(n_train + n_val, n))

        processed_ds = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        })
    else:
        print("No 'test' split found in the dataset. Proceeding with available splits.")
        # Shuffle and preprocess the first 1000 rows of each split if already split
        processed_ds = {}
        for split in ds.keys():
            shuffled = ds[split].shuffle(seed=42)
            small_split = shuffled.select(range(min(1000, len(shuffled))))
            processed_ds[split] = small_split.map(
                all_pairs,
                batched=True,
                remove_columns=small_split.column_names,
            )

            # Test access to the first 10 items of each split and print embedding shapes
for split in ["train", "validation", "test"]:
    print(f"\nChecking {split} split:")
    for i in range(min(5, len(processed_ds[split]))):
        item = processed_ds[split][i]
        # Convert lists back to tensors (HuggingFace datasets converts tensors to lists)
        img_embed = torch.tensor(item["image_embed"]) if isinstance(item["image_embed"], list) else item["image_embed"]
        text_embed = torch.tensor(item["text_embed"]) if isinstance(item["text_embed"], list) else item["text_embed"]
        
        print(f"Item {i}: image_embed shape = {img_embed.shape}, text_embed shape = {text_embed.shape}")
        print(f"Item {i}: image_embed dims = {img_embed.shape[1]}, text_embed dims = {text_embed.shape[1]}")
        print(f"Item {i}: caption = '{item['caption'][:50]}...'")  # Show first 50 chars of caption
# Print the size of each split
for split in ds.keys():
    print(f"{split.capitalize()} set size:", len(processed_ds[split]))

# Save the processed splits to pickle
with open("flickr30k_processed_splits.pkl", "wb") as f:
    pickle.dump(processed_ds, f)

def convert_to_tensors(item):
    """
    Helper function to convert dataset items back to tensors.
    HuggingFace datasets automatically converts tensors to lists during serialization.
    Use this function when loading items for training.
    """
    img_embed = torch.tensor(item["image_embed"]) if isinstance(item["image_embed"], list) else item["image_embed"]
    text_embed = torch.tensor(item["text_embed"]) if isinstance(item["text_embed"], list) else item["text_embed"]
    
    return {
        "image": item["image"],
        "caption": item["caption"],
        "img_id": item["img_id"],
        "sentids": item["sentids"],
        "image_embed": img_embed,
        "text_embed": text_embed
    }

def collate_fn(batch):
    """
    Custom collate function for DataLoader to properly stack tensors.
    Use this with PyTorch DataLoader.
    """
    # Convert all items to tensors first
    batch = [convert_to_tensors(item) for item in batch]
    
    # Stack tensors
    image_embeds = torch.stack([item["image_embed"] for item in batch])
    text_embeds = torch.stack([item["text_embed"] for item in batch])
    
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "img_ids": [item["img_id"] for item in batch],
        "sentids": [item["sentids"] for item in batch],
        "image_embeds": image_embeds,  # [batch_size, num_patches, hidden_dim]
        "text_embeds": text_embeds     # [batch_size, max_length, hidden_dim]
    }
