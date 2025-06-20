import logging
from typing import Union, List
from models.backbones import get_backbone
from models.classifiers import get_classifier

logger = logging.getLogger(__name__)

def build_model(
    backbone_name: str, 
    num_classes: int, 
    freeze_backbone: bool = False,
    freeze_blocks: Union[int, List[int], str] = None,
    **kwargs
):
    """
    Build a complete model with backbone and classifier.
    
    Args:
        backbone_name: Name of the backbone architecture
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the entire backbone
        freeze_blocks: Specific blocks to freeze:
            - int: Number of blocks to freeze from the start
            - list: Specific block indices to freeze
            - str: "all" to freeze all, "none" to freeze none
        **kwargs: Additional arguments passed to backbone and classifier
    
    Returns:
        Assembled model with backbone and classifier
    
    Raises:
        ValueError: If backbone_name is not supported or configuration is invalid
    """
    try:
        # Process freeze_blocks parameter
        freeze_params = {
            "freeze": freeze_backbone,
            "freeze_blocks": freeze_blocks
        }
        
        # Get backbone and its feature dimension
        backbone, feature_dim = get_backbone(backbone_name, **freeze_params)
        
        # Build classifier on top of backbone
        model = get_classifier(backbone_name, backbone, feature_dim, num_classes)
        
        logger.info(f"Built model with {backbone_name} backbone and {num_classes} output classes")
        return model
        
    except Exception as e:
        logger.error(f"Failed to build model: {str(e)}")
        raise ValueError(f"Model construction failed: {str(e)}")
