import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm

def load_flickr_data(split='train'):
    ds = load_dataset("nlphuji/flickr30k", cache_dir='data/flickr_data', split=split)
    return ds

def create_dataset_splits(raw_dataset, train_ratio=0.9, val_ratio=0.1, seed=42):
    """Create consistent train/val/test splits and save them."""
    # Get stable, reproducible splits
    n = len(raw_dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_train = int(n * train_ratio)

    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:] 
    
    splits = {
        'train': raw_dataset.iloc[train_indices],
        'validation': raw_dataset.iloc[val_indices], 
    }
    
    return splits



class FlickrDataset(Dataset):
    def __init__(self, split='train'):
        SAVE_DIR = Path('data', 'flickr_processed')
        IMG_SAVE_FILE = Path(SAVE_DIR, 'img_data.parquet')
        
        if split == 'train':
            CAPTIONS_SAVE_FILE = Path(SAVE_DIR, 'train_captions.parquet')
        elif split == 'validation':
            CAPTIONS_SAVE_FILE = Path(SAVE_DIR, 'val_captions.parquet')
        else:
            raise ValueError(f"Unsupported split: {split}. Use 'train' or 'validation'.")
            
        # Load everything as numpy arrays
        image_df = pd.read_parquet(IMG_SAVE_FILE)
        captions_df = pd.read_parquet(CAPTIONS_SAVE_FILE)        

        # Save the length of the dataset
        self.length = len(captions_df)
        self.split = split
        self.image_data = image_df['image'].to_numpy()
        self.captions_data = captions_df['caption'].tolist()
        self.image_ids = captions_df['image_id'].tolist()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        flat_image = self.image_data[self.image_ids[idx]]
        return {
            'image': torch.from_numpy(flat_image.reshape(3, 224, 224)),  # Convert to tensor here
            'caption': self.captions_data[idx]                               # Python string
        }


def qwen_collate_fn(batch, tokenizer):
    if len(batch) == 0:
        return None  # Return None instead of tuple
    
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    # Tokenize the captions (generates input_ids and attention_mask)
    tokenized = tokenizer(
        captions, 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
        max_length=512,  # Add max_length for safety
        return_attention_mask=True
    )
    
    # Add the image end token to signify the start of the text
    im_end_id = tokenizer.get_added_vocab()['<|im_end|>']
    input_ids = torch.cat([torch.full((len(tokenized['input_ids']), 1), im_end_id), tokenized['input_ids']], dim=-1)
    
    extended_attention_mask = torch.cat([torch.full((len(tokenized['attention_mask']), 1), 1), tokenized['attention_mask']], dim=-1)

    labels = input_ids.clone()
    # Shift the labels to the left by one
    labels[:, :-1] = labels[:, 1:]
    # Last token is the pad token (with safety check)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels[:, -1] = pad_token_id

    return {
        'images': torch.stack(images),  # Images are already tensors from __getitem__
        'input_ids': input_ids, 
        'attention_mask': extended_attention_mask, 
        'labels': labels
    }
    


def preprocess_flickr_data():
    '''
    keys: 
    'image', 
    'caption',
    'sentids': ['14795', '14796', '14797', '14798', '14799'],
    #'split': 'train',
    'img_id': '2959',
    'filename': '2042058437.jpg'}
     
    '''
    SAVE_DIR = Path('data', 'flickr_processed')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    IMG_SAVE_FILE = Path('img_data.parquet')
    CAPTIONS_TRAIN_FILE = 'train_captions.parquet'
    CAPTIONS_VAL_FILE = 'val_captions.parquet'



    ds = load_dataset("nlphuji/flickr30k", cache_dir='data/flickr_data', split='test', streaming=True)
    
    # Initialize variables for streaming processing
    processed_images = []
    all_captions = []
    batch_images = []
    batch_size = 32

    # Get image preprocessor and device
    print("Loading image processor...")
    image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")



    for idx, item in enumerate(tqdm(ds, desc="Streaming dataset")):
        # Collect images for batch processing
        batch_images.append(item['image'])
        
        # Process captions immediately (lightweight)
        for caption in item['caption']:
            all_captions.append({'caption': caption, 'image_id': idx})
        
        # Process image batch when full
        if len(batch_images) == batch_size:
            with torch.no_grad():
                batch_processed = image_processor(batch_images, return_tensors="pt")['pixel_values']
                batch_processed = batch_processed.cpu().numpy()
                flattened_batch = [img.flatten() for img in batch_processed]
                processed_images.extend(flattened_batch)
            
            batch_images.clear()  # Free memory immediately
            
    # Process remaining images
    if batch_images:
        with torch.no_grad():
            batch_processed = image_processor(batch_images, return_tensors="pt")['pixel_values']
            batch_processed = batch_processed.cpu().numpy()
            flattened_batch = [img.flatten() for img in batch_processed]
            processed_images.extend(flattened_batch)

    # Convert processed data to DataFrames (memory efficient - only metadata)
    print("Creating DataFrames from processed data...")
    images = pd.DataFrame({'image': processed_images})
    captions = pd.DataFrame(all_captions)  # Already in correct format
    
    print(f"Processed {len(images)} images and {len(captions)} captions")
    
    # Create splits from captions DataFrame
    print("Creating dataset splits...")
    splits = create_dataset_splits(captions, train_ratio=0.9, val_ratio=0.1, seed=42)

    train_captions_data = pd.DataFrame({
        'caption': splits['train']['caption'],
        'image_id': splits['train']['image_id'],
    })
    val_captions_data = pd.DataFrame({
        'caption': splits['validation']['caption'],
        'image_id': splits['validation']['image_id'],
    })

    # Save with consistent indexing
    print("Saving preprocessed data...")
    images.to_parquet(SAVE_DIR / IMG_SAVE_FILE, index=False)
    train_captions_data.to_parquet(SAVE_DIR / CAPTIONS_TRAIN_FILE, index=False)
    val_captions_data.to_parquet(SAVE_DIR / CAPTIONS_VAL_FILE, index=False)
    
    print(f"✅ Dataset preprocessing completed!")
    print(f"   📁 Images saved: {len(images)} processed images")
    print(f"   📝 Train captions: {len(train_captions_data)}")
    print(f"   📝 Val captions: {len(val_captions_data)}")

    return ds


    


def create_hf_dataset_for_upload(data_dir='data/flickr_processed'):
    """Convert preprocessed data to Hugging Face Dataset format for upload."""
    from datasets import Dataset, DatasetDict
    import json
    
    data_dir = Path(data_dir)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    images_df = pd.read_parquet(data_dir / 'img_data.parquet')
    train_captions_df = pd.read_parquet(data_dir / 'train_captions.parquet')
    val_captions_df = pd.read_parquet(data_dir / 'val_captions.parquet')
    
    # Create mapping from image_id to image data
    image_id_to_data = {idx: img_data for idx, img_data in enumerate(images_df['image'])}
    
    def prepare_split_data(captions_df, split_name):
        print(f"Preparing {split_name} split...")
        data = []
        for _, row in captions_df.iterrows():
            data.append({
                'image_id': row['image_id'],
                'image': image_id_to_data[row['image_id']],  # CLIP-processed image tensor
                'caption': row['caption'],
                'split': split_name
            })
        return data
    
    # Prepare data for both splits
    train_data = prepare_split_data(train_captions_df, 'train')
    val_data = prepare_split_data(val_captions_df, 'validation')
    
    # Create HF Datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    # Add metadata
    dataset_dict.info.description = """
    Flickr30k dataset preprocessed with CLIP ViT-Large-Patch14 image processor.
    
    Original dataset: https://huggingface.co/datasets/nlphuji/flickr30k
    Preprocessing: Images processed with openai/clip-vit-large-patch14 processor
    
    Each image is a tensor of shape [3, 224, 224] with CLIP normalization applied.
    Ready for use with CLIP-based vision-language models.
    """
    
    return dataset_dict

def upload_to_hf_hub(dataset_dict, repo_name, private=True):
    """Upload dataset to Hugging Face Hub."""
    from huggingface_hub import HfApi
    
    print(f"Uploading dataset to {repo_name}...")
    
    # Upload dataset
    dataset_dict.push_to_hub(
        repo_name,
        private=private,  # Start private, make public later if desired
        commit_message="Initial upload of CLIP-preprocessed Flickr30k dataset"
    )
    
    print(f"✅ Dataset uploaded successfully!")
    print(f"🔗 Access at: https://huggingface.co/datasets/{repo_name}")
    
    return f"https://huggingface.co/datasets/{repo_name}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the Flickr30k dataset')
    parser.add_argument('--upload', type=str, help='Upload preprocessed dataset to HF Hub (provide repo name)')
    parser.add_argument('--private', action='store_true', help='Make uploaded dataset private (default: True)')
    args = parser.parse_args()
    
    if args.preprocess:
        print("🚀 Starting Flickr30k dataset preprocessing...")
        preprocess_flickr_data()
    elif args.upload:
        print(f"📤 Preparing dataset for upload to {args.upload}...")
        dataset_dict = create_hf_dataset_for_upload()
        upload_to_hf_hub(dataset_dict, args.upload, private=not args.private)
    else:
        print("Available commands:")
        print("  --preprocess: Preprocess the Flickr30k dataset")
        print("  --upload <repo-name>: Upload preprocessed dataset to HF Hub")
        print("\nExamples:")
        print("  python dataset.py --preprocess")
        print("  python dataset.py --upload your-username/flickr30k-clip-processed --private")
