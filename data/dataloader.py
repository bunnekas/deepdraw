import os
import glob
import tarfile
import io
import logging
from typing import List, Optional
import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

logger = logging.getLogger(__name__)

def validate_data_path(path: str) -> None:
    """
    Validate that the data path exists and is accessible.
    
    Args:
        path: Path to validate
        
    Raises:
        FileNotFoundError: If path doesn't exist
        PermissionError: If path isn't accessible
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path does not exist: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Data path is not a directory: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Data path is not readable: {path}")

def load_categories(categories_file: str) -> List[str]:
    """
    Load category labels from file.
    
    Args:
        categories_file: Path to categories file
        
    Returns:
        List of category labels
        
    Raises:
        FileNotFoundError: If categories file doesn't exist
        ValueError: If categories file is empty
    """
    if not os.path.exists(categories_file):
        raise FileNotFoundError(f"Categories file not found: {categories_file}")
        
    with open(categories_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
        
    if not classes:
        raise ValueError(f"Categories file is empty: {categories_file}")
        
    logger.info(f"Loaded {len(classes)} categories from {categories_file}")
    return classes

def get_loader(
    tar_dir: str, 
    batch_size: int = 64, 
    num_workers: int = 4, 
    shuffle: bool = False, 
    shuffle_buffer: int = 1000, 
    categories_file: Optional[str] = None,
    image_size: int = 224,
    normalize: bool = True,
    prefetch_factor: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a data loader for WebDataset tar files.
    
    Args:
        tar_dir: Directory containing tar files
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        shuffle_buffer: Size of shuffle buffer
        categories_file: Path to categories file (if None, uses default)
        image_size: Size to resize images to
        normalize: Whether to normalize images with ImageNet stats
        
    Returns:
        DataLoader for the dataset
        
    Raises:
        FileNotFoundError: If tar directory or files don't exist
        ValueError: If categories file is invalid
    """
    try:
        # Validate data directory
        validate_data_path(tar_dir)
        
        # Find all tar files in the directory
        tar_files = sorted(glob.glob(os.path.join(tar_dir, "data-*.tar")))
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in {tar_dir}")
            
        logger.info(f"Found {len(tar_files)} tar files in {tar_dir}")

        # WebDataset pattern
        pattern = os.path.join(tar_dir, "data-{00000..%05d}.tar" % (len(tar_files)-1))

        # Set default categories file if not provided
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if categories_file is None:
            categories_file = os.path.join(base_dir, "categories_traintest.txt")

        # Load categories
        classes = load_categories(categories_file)
        label2idx = {label: idx for idx, label in enumerate(classes)}

        # Build transformation pipeline
        transform_list = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
            
        transform = transforms.Compose(transform_list)

        def my_decoder(key, data):
            try:
                if key.endswith(".jpg"):
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    return transform(img)
                elif key.endswith(".cls"):
                    label = data.decode("utf-8").strip()
                    if label not in label2idx:
                        logger.warning(f"Unknown label encountered: {label}")
                        return None
                    return label
                return data
            except Exception as e:
                logger.warning(f"Skipping corrupted {key}: {e}")
                return None

        dataset = wds.WebDataset(pattern, shardshuffle=100 if shuffle else False).decode(my_decoder)

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = (
            dataset
            .to_tuple("jpg", "cls")
            .map_tuple(lambda img: img, lambda cls: label2idx[cls])
            .batched(batch_size)
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory
        )
        
        logger.info(f"Created dataloader with batch_size={batch_size}, num_workers={num_workers}")
        return loader
        
    except Exception as e:
        logger.error(f"Failed to create dataloader: {str(e)}")
        raise


def count_samples(folder: str) -> int:
    """
    Count the number of samples in tar files.
    
    Args:
        folder: Directory containing tar files
        
    Returns:
        Number of samples
        
    Raises:
        FileNotFoundError: If folder doesn't exist
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
        
    count = 0
    tar_files = sorted(glob.glob(os.path.join(folder, "data-*.tar")))
    
    if not tar_files:
        logger.warning(f"No tar files found in {folder}")
        return 0
        
    for tar_path in tar_files:
        try:
            with tarfile.open(tar_path, "r") as tar:
                file_count = sum(1 for member in tar if member.name.endswith(".jpg"))
                count += file_count
                logger.debug(f"Counted {file_count} samples in {os.path.basename(tar_path)}")
        except Exception as e:
            logger.warning(f"Could not read {tar_path} - {str(e)}")
            continue
            
    logger.info(f"Total samples in {folder}: {count}")
    return count