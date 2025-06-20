import torch
import logging
from typing import Tuple, Union, List

logger = logging.getLogger(__name__)

def load_dinov2(
    freeze: bool = False,
    freeze_blocks: Union[int, List[int], str] = None,
    **kwargs
) -> Tuple[torch.nn.Module, int]:
    """
    Load DinoV2 backbone with configurable freezing options.
    
    Args:
        freeze: Whether to freeze the entire backbone
        freeze_blocks: Specific blocks to freeze:
            - int: Number of blocks to freeze from the start
            - list: Specific block indices to freeze
            - str: "all" to freeze all, "none" to freeze none
        **kwargs: Additional arguments passed to the backbone
        
    Returns:
        Tuple of (backbone, feature_dimension)
    """
    try:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        feature_dim = 1024
        
        # Handle freezing logic
        if freeze:
            logger.info("Freezing entire DinoV2 backbone")
            for param in backbone.parameters():
                param.requires_grad = False
        elif freeze_blocks is not None:
            if isinstance(freeze_blocks, int):
                # Freeze the first n blocks
                block_indices = list(range(freeze_blocks))
                logger.info(f"Freezing first {freeze_blocks} blocks of DinoV2 backbone")
            elif isinstance(freeze_blocks, list):
                # Freeze specific blocks
                block_indices = freeze_blocks
                logger.info(f"Freezing blocks {block_indices} of DinoV2 backbone")
            elif freeze_blocks == "all":
                # Freeze all blocks
                block_indices = list(range(24))  # DinoV2 has 24 blocks
                logger.info("Freezing all blocks of DinoV2 backbone")
            elif freeze_blocks == "none":
                # Don't freeze any blocks
                block_indices = []
                logger.info("Not freezing any blocks of DinoV2 backbone")
            else:
                raise ValueError(f"Invalid freeze_blocks value: {freeze_blocks}")
                
            # Apply freezing to specified blocks
            for name, param in backbone.named_parameters():
                should_freeze = any(f'blocks.{i}' in name for i in block_indices)
                if should_freeze:
                    param.requires_grad = False
        else:
            logger.info("DinoV2 backbone is fully trainable")
            
        # Log parameter status
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in backbone.parameters())
        logger.info(f"DinoV2 backbone: {trainable_params:,} trainable parameters out of {total_params:,} total")
            
        return backbone, feature_dim
        
    except Exception as e:
        logger.error(f"Failed to load DinoV2 backbone: {str(e)}")
        raise RuntimeError(f"DinoV2 backbone loading failed: {str(e)}")
