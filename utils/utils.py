import os
import random
import logging
import yaml
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)

def seed_all(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed} for all libraries")

def load_config(config_path: str, base_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional base config.
    
    Args:
        config_path: Path to config file
        base_config_path: Optional path to base config file
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load base config if provided
    config = {}
    if base_config_path is not None:
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(f"Base config file not found: {base_config_path}")
            
        with open(base_config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                logger.info(f"Loaded base config from {base_config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid base config file: {str(e)}")
    
    # Load main config
    with open(config_path, 'r') as f:
        try:
            main_config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid config file: {str(e)}")
    
    # Deep merge configs
    config = deep_merge(config, main_config)
    
    # Convert string values that should be numeric
    if 'train' in config:
        for key in ['weight_decay', 'backbone_lr', 'classifier_lr']:
            if key in config['train'] and isinstance(config['train'][key], str):
                config['train'][key] = float(config['train'][key])
    
    return config

def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, with override_dict taking precedence.
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with overrides
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def get_device() -> torch.device:
    """
    Get the appropriate device for PyTorch operations.
    
    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    return device

def setup_gpu_environment() -> None:
    """
    Configure GPU environment for optimal performance.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.max_split_size_mb = 128
        torch.backends.cudnn.benchmark = True
        logger.info("GPU environment configured for optimal performance")
    else:
        logger.info("No GPU available, using CPU")

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {trainable:,} trainable parameters out of {total:,} total")
    return trainable, total

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary for required fields and valid values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['data', 'train', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Check data paths
    for key in ['train_dir', 'test_dir']:
        if key not in config['data']:
            raise ValueError(f"Missing required data path: {key}")
    
    # Check model configuration
    if 'type' not in config['model']:
        raise ValueError("Missing model type in config")
    if 'num_classes' not in config['model']:
        raise ValueError("Missing num_classes in model config")
    
    # Check training parameters
    required_train_params = ['batch_size', 'epochs', 'backbone_lr', 'classifier_lr']
    for param in required_train_params:
        if param not in config['train']:
            raise ValueError(f"Missing required training parameter: {param}")
    
    logger.info("Configuration validated successfully")
