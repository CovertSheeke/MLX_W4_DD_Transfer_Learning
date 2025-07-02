import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import pyarrow.parquet as pq


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


class DatasetCacheError(Exception):
    """Custom exception for dataset cache/download errors."""
    pass


class FlickrDatasetCache:
    """Handles caching and downloading of preprocessed Flickr30k dataset files."""
    
    def __init__(self, cache_dir='data/flickr_processed', remote_repo=None, force_download=False):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Local directory to store cached files
            remote_repo: Remote repository to download from (e.g., HF Hub repo name)
            force_download: If True, always download fresh data (skip cache)
        """
        self.cache_dir = Path(cache_dir)
        self.remote_repo = remote_repo
        self.force_download = force_download
        
        # Define expected cache files
        self.required_files = {
            'images': self.cache_dir / 'img_data.parquet',
            'train_captions': self.cache_dir / 'train_captions.parquet', 
            'val_captions': self.cache_dir / 'val_captions.parquet'
        }
    
    def cache_exists(self) -> bool:
        """Check if all required cache files exist and are valid."""
        if self.force_download:
            return False
            
        # Check if all files exist
        for file_type, file_path in self.required_files.items():
            if not file_path.exists():
                print(f"❌ Cache file missing: {file_path}")
                return False
                
        # Basic validation - check if files are readable using metadata (memory efficient)
        try:
            for file_type, file_path in self.required_files.items():
                # Use parquet metadata instead of loading entire file
                parquet_file = pq.ParquetFile(file_path)
                if parquet_file.metadata.num_rows == 0:
                    print(f"❌ Cache file empty: {file_path}")
                    return False
            print("✅ Valid cache found")
            return True
        except Exception as e:
            print(f"❌ Cache validation failed: {e}")
            return False
    
    def download_from_remote(self):
        """Download dataset files from remote repository."""
        if not self.remote_repo:
            raise DatasetCacheError(
                "No cached data found and no remote_repo specified. "
                "Either run preprocessing locally or provide a remote_repo."
            )
        
        print(f"📥 Downloading from remote repository: {self.remote_repo}")
        
        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try HuggingFace Hub download
            self._download_from_hf_hub()
            
        except Exception as e:
            raise DatasetCacheError(f"Failed to download from remote repository: {e}")
    
    def _download_from_hf_hub(self):
        """Download from HuggingFace Hub repository."""
        try:
            from huggingface_hub import snapshot_download
            
            print(f"Downloading from HuggingFace Hub: {self.remote_repo}")
            snapshot_download(
                repo_id=self.remote_repo,
                repo_type="dataset",
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False  # Copy files instead of symlinks
            )
            print("✅ Download completed")
            
        except ImportError:
            raise DatasetCacheError(
                "huggingface_hub not available. Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise DatasetCacheError(f"HuggingFace Hub download failed: {e}")
    
    def ensure_data_available(self):
        """
        Ensure dataset files are available locally.
        
        This is the main method called by FlickrDataset.__init__().
        It handles the cache/download logic transparently.
        """
        if not self.cache_exists():
            if self.remote_repo:
                print("📋 Cache not found, attempting download...")
                self.download_from_remote()
                
                # Verify download worked
                if not self.cache_exists():
                    raise DatasetCacheError("Download completed but cache validation failed")
            else:
                raise DatasetCacheError(
                    f"No cached data found in {self.cache_dir} and no remote_repo specified.\n"
                    f"Please either:\n"
                    f"  1. Run preprocessing: python dataset.py --preprocess\n"
                    f"  2. Provide remote_repo parameter for download"
                )
        
        print(f"📁 Using cached data from: {self.cache_dir}")
    
    def get_file_paths(self) -> dict:
        """Return dictionary of file paths for the dataset."""
        return self.required_files.copy()


class FlickrDataset(Dataset):
    # Class-level shared image cache
    _shared_image_cache = {}
    
    def __init__(self, split='train', remote_repo=None, cache_dir='data/flickr_processed', force_download=False):
        """
        Initialize Flickr30k dataset.
        
        Args:
            split: Dataset split ('train' or 'validation')
            remote_repo: Remote repository to download from if cache missing (e.g., 'username/repo-name')
            cache_dir: Local cache directory
            force_download: If True, always download fresh data
        """
        # Validate split
        if split not in ['train', 'validation']:
            raise ValueError(f"Unsupported split: {split}. Use 'train' or 'validation'.")
        
        # Setup cache manager and ensure data is available
        cache_manager = FlickrDatasetCache(cache_dir, remote_repo, force_download)
        cache_manager.ensure_data_available()
        
        # Get file paths
        file_paths = cache_manager.get_file_paths()
        
        # Determine caption file based on split
        if split == 'train':
            captions_file = file_paths['train_captions']
        else:  # validation
            captions_file = file_paths['val_captions']
            
        # Load captions 
        captions_df = pd.read_parquet(captions_file)        
        
        # Load or reuse shared image data
        images_file_path = str(file_paths['images'])  # Convert to string for dict key
        
        if images_file_path not in self._shared_image_cache:
            print(f"Streaming image data from {file_paths['images']} (first time)...")
            
            # Stream parquet file to build numpy array incrementally
            parquet_file = pq.ParquetFile(file_paths['images'])
            total_rows = parquet_file.metadata.num_rows
            print(f"Total images to load: {total_rows}")
            
            # Get the shape of a single image by reading the first batch
            first_batch = next(parquet_file.iter_batches(batch_size=1))
            first_df = first_batch.to_pandas()
            image_shape = first_df['image'].iloc[0].shape
            print(f"Image shape: {image_shape}")
            
            # Pre-allocate the entire numpy array
            print(f"Pre-allocating numpy array for {total_rows} images...")
            image_data = np.empty((total_rows,) + image_shape, dtype=first_df['image'].iloc[0].dtype)
            
            # Stream and write directly into the pre-allocated array
            chunk_size = 1000  # Process in chunks to manage memory
            current_idx = 0
            
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                # Convert batch to pandas and extract images
                batch_df = batch.to_pandas()
                batch_size_actual = len(batch_df)
                
                # Write directly into the pre-allocated array
                for i, image_array in enumerate(batch_df['image']):
                    image_data[current_idx + i] = image_array
                
                current_idx += batch_size_actual
                
                # Print progress
                if current_idx % (chunk_size * 10) == 0 or current_idx == total_rows:
                    print(f"  Loaded {current_idx}/{total_rows} images...")
            
            print(f"✅ Loaded {len(image_data)} images into pre-allocated numpy array")
            
            # Cache the image data for reuse
            self._shared_image_cache[images_file_path] = image_data
        else:
            print(f"✅ Reusing cached image data ({len(self._shared_image_cache[images_file_path])} images)")
        
        # Reference the shared image data
        self.image_data = self._shared_image_cache[images_file_path]

        # Save dataset attributes
        self.length = len(captions_df)
        self.split = split
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


    


def upload_parquet_files_to_hf_hub(data_dir='data/flickr_processed', repo_name=None, private=True):
    """
    Upload parquet files directly to HuggingFace Hub without loading into memory.
    This is much more memory efficient for large datasets.
    """
    from huggingface_hub import HfApi, create_repo
    import tempfile
    import os
    
    if not repo_name:
        raise ValueError("Please provide a repo_name for upload")
    
    data_dir = Path(data_dir)
    
    # Check that all required files exist
    required_files = {
        'img_data.parquet': data_dir / 'img_data.parquet',
        'train_captions.parquet': data_dir / 'train_captions.parquet', 
        'val_captions.parquet': data_dir / 'val_captions.parquet'
    }
    
    for filename, filepath in required_files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository
    try:
        create_repo(repo_name, repo_type="dataset", private=private)
        print(f"✅ Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Create a README file with dataset information
    readme_content = f"""# Flickr30k CLIP-Preprocessed Dataset

This dataset contains the Flickr30k dataset preprocessed with CLIP ViT-Large-Patch14 image processor.

## Files

- `img_data.parquet`: Preprocessed images as flattened numpy arrays (shape: [3, 224, 224] -> flattened)
- `train_captions.parquet`: Training split captions with image_id mapping
- `val_captions.parquet`: Validation split captions with image_id mapping

## Usage

```python
import pandas as pd
import torch
import numpy as np

# Load the data
images_df = pd.read_parquet('img_data.parquet')
train_captions_df = pd.read_parquet('train_captions.parquet')
val_captions_df = pd.read_parquet('val_captions.parquet')

# Access an image
image_id = 0
flat_image = images_df.iloc[image_id]['image']
image_tensor = torch.from_numpy(flat_image.reshape(3, 224, 224))

# Access captions for that image
captions_for_image = train_captions_df[train_captions_df['image_id'] == image_id]['caption'].tolist()
```

## Original Dataset

Original dataset: [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)

## Preprocessing

Images were processed using the CLIP ViT-Large-Patch14 image processor:
- Resized to 224x224
- CLIP normalization applied
- Converted to tensors and flattened for storage efficiency

## Dataset Statistics

- Total images: Check `img_data.parquet` length
- Train captions: Check `train_captions.parquet` length  
- Validation captions: Check `val_captions.parquet` length
- Train/Validation split: 90/10
"""

    # Create temporary README file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme_content)
        readme_path = f.name
    
    try:
        # Upload README first
        print("📝 Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add dataset README"
        )
        
        # Upload each parquet file
        for filename, filepath in required_files.items():
            print(f"📤 Uploading {filename}... (this may take a while for large files)")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_name,
                repo_type="dataset",
                commit_message=f"Upload {filename}"
            )
            print(f"✅ {filename} uploaded successfully")
            
    finally:
        # Clean up temporary README file
        os.unlink(readme_path)
    
    print(f"\n🎉 Dataset upload completed!")
    print(f"🔗 Access at: https://huggingface.co/datasets/{repo_name}")
    print(f"\n📋 To use this dataset:")
    print(f"```python")
    print(f"from huggingface_hub import hf_hub_download")
    print(f"import pandas as pd")
    print(f"")
    print(f"# Download files")
    print(f"img_file = hf_hub_download(repo_id='{repo_name}', filename='img_data.parquet', repo_type='dataset')")
    print(f"train_file = hf_hub_download(repo_id='{repo_name}', filename='train_captions.parquet', repo_type='dataset')")
    print(f"val_file = hf_hub_download(repo_id='{repo_name}', filename='val_captions.parquet', repo_type='dataset')")
    print(f"")
    print(f"# Load data")
    print(f"images_df = pd.read_parquet(img_file)")
    print(f"train_captions_df = pd.read_parquet(train_file)")
    print(f"val_captions_df = pd.read_parquet(val_file)")
    print(f"```")
    
    return f"https://huggingface.co/datasets/{repo_name}"


def create_hf_dataset_for_upload_chunked(data_dir='data/flickr_processed', chunk_size=1000):
    """Convert preprocessed data to Hugging Face Dataset format using chunked processing."""
    from datasets import Dataset, DatasetDict
    import pyarrow.parquet as pq
    import tempfile
    import gc
    
    data_dir = Path(data_dir)
    
    # Load caption data (this is much smaller)
    print("Loading caption data...")
    train_captions_df = pd.read_parquet(data_dir / 'train_captions.parquet')
    val_captions_df = pd.read_parquet(data_dir / 'val_captions.parquet')
    
    # Get unique image IDs needed for each split
    train_image_ids = set(train_captions_df['image_id'].unique())
    val_image_ids = set(val_captions_df['image_id'].unique())
    all_needed_ids = train_image_ids.union(val_image_ids)
    
    print(f"Need to load {len(all_needed_ids)} unique images")
    
    # Read parquet file in chunks to avoid memory issues
    parquet_file = pq.ParquetFile(data_dir / 'img_data.parquet')
    total_rows = parquet_file.metadata.num_rows
    
    print(f"Reading {total_rows} images in chunks of {chunk_size}...")
    
    # Store image data as we process chunks
    image_id_to_data = {}
    
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        print(f"Processing chunk {batch_idx + 1}/{(total_rows + chunk_size - 1) // chunk_size}")
        
        # Convert batch to pandas for easier processing
        chunk_df = batch.to_pandas()
        
        # Only keep images we actually need
        for idx, row in chunk_df.iterrows():
            global_idx = batch_idx * chunk_size + idx
            if global_idx in all_needed_ids:
                image_id_to_data[global_idx] = row['image']
        
        # Free memory
        del chunk_df
        gc.collect()
    
    print(f"Loaded {len(image_id_to_data)} required images")
    
    def prepare_split_data(captions_df, split_name):
        print(f"Preparing {split_name} split...")
        data = []
        for _, row in captions_df.iterrows():
            image_id = row['image_id']
            if image_id in image_id_to_data:
                data.append({
                    'image_id': image_id,
                    'image': image_id_to_data[image_id],
                    'caption': row['caption'],
                    'split': split_name
                })
            else:
                print(f"Warning: Missing image data for image_id {image_id}")
        return data
    
    # Prepare data for both splits
    train_data = prepare_split_data(train_captions_df, 'train')
    val_data = prepare_split_data(val_captions_df, 'validation')
    
    # Create HF Datasets
    print("Creating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Free the image cache
    del image_id_to_data
    gc.collect()
    
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

def create_hf_dataset_for_upload(data_dir='data/flickr_processed'):
    """Wrapper that uses chunked processing to avoid memory issues."""
    return create_hf_dataset_for_upload_chunked(data_dir, chunk_size=1000)

def upload_to_hf_hub_streaming(data_dir, repo_name, private=True, upload_batch_size=500):
    """Upload dataset to HF Hub in streaming chunks to minimize memory usage."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, create_repo
    import pyarrow.parquet as pq
    import gc
    
    data_dir = Path(data_dir)
    
    # Create the repository first
    api = HfApi()
    try:
        create_repo(repo_name, repo_type="dataset", private=private)
        print(f"✅ Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Load caption data
    print("Loading caption data...")
    train_captions_df = pd.read_parquet(data_dir / 'train_captions.parquet')
    val_captions_df = pd.read_parquet(data_dir / 'val_captions.parquet')
    
    # Get unique image IDs for each split
    train_image_ids = set(train_captions_df['image_id'].unique())
    val_image_ids = set(val_captions_df['image_id'].unique())
    
    # Process each split separately
    for split_name, captions_df, needed_ids in [
        ('train', train_captions_df, train_image_ids),
        ('validation', val_captions_df, val_image_ids)
    ]:
        print(f"\n📤 Processing {split_name} split...")
        
        # Read images in chunks and match with captions
        parquet_file = pq.ParquetFile(data_dir / 'img_data.parquet')
        total_rows = parquet_file.metadata.num_rows
        
        split_data = []
        processed_images = 0
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=1000)):
            chunk_df = batch.to_pandas()
            
            # Find matching captions for images in this chunk
            for idx, row in chunk_df.iterrows():
                global_idx = batch_idx * 1000 + idx
                if global_idx in needed_ids:
                    # Get all captions for this image
                    image_captions = captions_df[captions_df['image_id'] == global_idx]
                    for _, caption_row in image_captions.iterrows():
                        split_data.append({
                            'image_id': global_idx,
                            'image': row['image'],
                            'caption': caption_row['caption'],
                            'split': split_name
                        })
                        processed_images += 1
                
                # Upload in batches to avoid memory buildup
                if len(split_data) >= upload_batch_size:
                    print(f"  Uploading batch of {len(split_data)} items...")
                    batch_dataset = Dataset.from_list(split_data)
                    
                    # Upload this batch
                    batch_dataset.push_to_hub(
                        repo_name,
                        split=split_name,
                        private=private,
                        append=True if processed_images > upload_batch_size else False
                    )
                    
                    # Clear memory
                    del split_data, batch_dataset
                    split_data = []
                    gc.collect()
            
            del chunk_df
            gc.collect()
        
        # Upload remaining data
        if split_data:
            print(f"  Uploading final batch of {len(split_data)} items...")
            batch_dataset = Dataset.from_list(split_data)
            batch_dataset.push_to_hub(
                repo_name,
                split=split_name,
                private=private,
                append=processed_images > len(split_data)
            )
            del split_data, batch_dataset
            gc.collect()
        
        print(f"✅ {split_name} split uploaded: {processed_images} items")
    
    print(f"\n🎉 Dataset upload completed!")
    print(f"🔗 Access at: https://huggingface.co/datasets/{repo_name}")
    return f"https://huggingface.co/datasets/{repo_name}"

def upload_to_hf_hub(dataset_dict, repo_name, private=True):
    """Upload dataset to Hugging Face Hub (original method)."""
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
    parser.add_argument('--upload-parquet', type=str, help='Upload parquet files directly to HF Hub (memory efficient)')
    parser.add_argument('--private', action='store_true', help='Make uploaded dataset private (default: True)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for reading parquet files (default: 1000)')
    args = parser.parse_args()
    
    if args.preprocess:
        print("🚀 Starting Flickr30k dataset preprocessing...")
        preprocess_flickr_data()
    elif args.upload_parquet:
        print(f"📤 Uploading parquet files directly to {args.upload_parquet}...")
        upload_parquet_files_to_hf_hub(
            data_dir='data/flickr_processed',
            repo_name=args.upload_parquet, 
            private=not args.private
        )
    elif args.upload:
        print(f"📤 Preparing dataset for upload to {args.upload}...")
        print(f"Using chunk size: {args.chunk_size} (use --chunk-size to adjust if running out of memory)")
        dataset_dict = create_hf_dataset_for_upload_chunked(chunk_size=args.chunk_size)
        upload_to_hf_hub(dataset_dict, args.upload, private=not args.private)
    else:
        print("Available commands:")
        print("  --preprocess: Preprocess the Flickr30k dataset")
        print("  --upload <repo-name>: Upload preprocessed dataset to HF Hub (memory intensive)")
        print("  --upload-parquet <repo-name>: Upload parquet files directly to HF Hub (memory efficient)")
        print("  --chunk-size <size>: Control memory usage when uploading (default: 1000)")
        print("\nExamples:")
        print("  python dataset.py --preprocess")
        print("  python dataset.py --upload-parquet your-username/flickr30k-clip-processed --private")
        print("  python dataset.py --upload your-username/flickr30k-clip-processed --private")
        print("  python dataset.py --upload your-username/flickr30k-clip-processed --chunk-size 500  # for low memory")
