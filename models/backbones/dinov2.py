import os
import torch
import torch.nn as nn
import logging
from typing import Tuple, Union, List
from utils.utils import count_trainable_parameters, get_block_parameter_count

logger = logging.getLogger(__name__)

def load_dinov2(freeze_blocks: Union[None, str, int, List[int], Tuple[int, int]] = None, **kwargs) -> Tuple[nn.Module, int]:
    """
    Load DinoV2 backbone with configurable freezing options.
    
    Args:
        freeze_blocks: Controls which transformer blocks to freeze:
            - None: Don't freeze any blocks (all trainable)
            - "all": Freeze all blocks
            - int: Freeze blocks from 0 to (freeze_blocks-1)
            - list: List of specific block indices to freeze
            - tuple: (start, end) to freeze blocks from start to end-1
        **kwargs: Additional arguments passed to the model
        
    Returns:
        Tuple of (backbone, feature_dimension)
    """
    try:
        # Check if running on CPU or GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # If on CPU, use a patched version of the model that doesn't use xformers
        if device.type == 'cpu' and os.environ.get("DINOV2_DISABLE_XFORMERS", "0") != "0":
            logger.info("Running on CPU: Using standard attention instead of xformers")
            # Set environment variable to disable xformers
            os.environ["DINOV2_DISABLE_XFORMERS"] = "1"
            
        # Load the model
        logger.info("Loading DinoV2 backbone...")
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        feature_dim = 1024
        
        # First, unfreeze all parameters to ensure we start from a clean state
        for param in backbone.parameters():
            param.requires_grad = True
        
        # Handle freezing based on the type of freeze_blocks
        if freeze_blocks is not None:
            if freeze_blocks == "all":
                # Freeze all blocks
                for name, param in backbone.named_parameters():
                    if 'blocks' in name:
                        param.requires_grad = False
                logger.info(f"Froze all blocks in DinoV2 backbone")
            
            elif isinstance(freeze_blocks, int):
                # Freeze blocks from 0 to (freeze_blocks-1)
                for name, param in backbone.named_parameters():
                    for i in range(freeze_blocks):
                        if f'blocks.{i}.' in name:
                            param.requires_grad = False
                logger.info(f"Froze blocks 0-{freeze_blocks-1} in DinoV2 backbone")
            
            elif isinstance(freeze_blocks, list):
                # Freeze specific block indices
                for name, param in backbone.named_parameters():
                    for i in freeze_blocks:
                        if f'blocks.{i}.' in name:
                            param.requires_grad = False
                logger.info(f"Froze blocks {freeze_blocks} in DinoV2 backbone")
            
            elif isinstance(freeze_blocks, tuple) and len(freeze_blocks) == 2:
                # Freeze blocks from start to end-1
                start, end = freeze_blocks
                for name, param in backbone.named_parameters():
                    for i in range(start, end):
                        if f'blocks.{i}.' in name:
                            param.requires_grad = False
                logger.info(f"Froze blocks {start}-{end-1} in DinoV2 backbone")
            else:
                logger.warning(f"Unrecognized freeze_blocks format: {freeze_blocks}. No blocks frozen.")
        
        # Count and log trainable parameters using the utility function
        from utils.utils import count_trainable_parameters
        trainable_params, total_params = count_trainable_parameters(backbone)
        logger.info(f"{trainable_params:,} trainable parameters out of {total_params:,} total")
        
        # Optional: Print detailed block-by-block breakdown
        if logger.isEnabledFor(logging.DEBUG):
            from utils.utils import get_block_parameter_count
            block_params = get_block_parameter_count(backbone)
            for block_name, param_count in block_params:
                logger.debug(f"{block_name}: {param_count:,} parameters")
        
        return backbone, feature_dim
        
    except Exception as e:
        logger.error(f"Failed to load DinoV2 backbone: {str(e)}")
        raise RuntimeError(f"DinoV2 backbone loading failed: {str(e)}")