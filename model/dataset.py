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
        return {
            'image': torch.from_numpy(self.image_data[self.image_ids[idx]]),  # Convert to tensor here
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
    

def get_optimal_batch_size(device, image_size=(224, 224)):
    """Automatically determine optimal batch size based on available memory."""
    if device.type == 'cuda':
        # GPU memory-based calculation
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory > 16e9:  # 16GB+
            return 64
        elif gpu_memory > 8e9:  # 8GB+
            return 32
        else:  # <8GB
            return 16
    else:
        # CPU - smaller batches to avoid RAM issues
        return 8
    

def process_image_batch(image_list, image_processor, batch_size=None):
    # Define device properly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if batch_size is None:
        batch_size = get_optimal_batch_size(device)

    processed_images = []
    total_batches = len(image_list) // batch_size
    with tqdm(total=total_batches, desc=f"Processing on {device}") as pbar:
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
        
            with torch.no_grad():
                batch_processed = image_processor(batch_images, return_tensors="pt")['pixel_values']

                batch_processed = batch_processed.cpu().numpy()
                processed_images.extend(batch_processed)
                pbar.update(1)

    return processed_images

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



    ds = load_dataset("nlphuji/flickr30k", cache_dir='data/flickr_data', split='test')
    images = pd.DataFrame({'image': ds['image']})
    captions = pd.DataFrame({'caption': ds['caption']}).explode('caption')
    captions = captions.reset_index().rename(columns={'index': 'image_id'})

    # Get image preprocessor
    print("Loading image processor...")
    image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Convert DataFrame column to list for batch processing
    print("Starting batch image processing...")
    processed_images = process_image_batch(images['image'].tolist(), image_processor)

    # Preprocess the images
    images['image'] = processed_images
    
    #Note: This is a dictionary with keys 'train' and 'validation' whose values are pandas dataframes
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


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the Flickr30k dataset')
    args = parser.parse_args()
    
    if args.preprocess:
        print("🚀 Starting Flickr30k dataset preprocessing...")
        preprocess_flickr_data()
    else:
        print("Use --preprocess flag to preprocess the dataset")
        print("Example: python dataset.py --preprocess")


