import os
import logging
from typing import Dict, Any, Optional
import wandb
import torch

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Centralized metrics logging for wandb tracking.
    """
    
    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        group: Optional[str] = None,
    ):
        """
        Initialize metrics logger.
        
        Args:
            project: Project name for wandb
            name: Run name
            config: Configuration dictionary
            group: Group name for wandb
            log_dir: Directory for local logs
        """
        self.use_wandb = True
        
        # Set wandb directory
        wandb_dir = "logs"
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        # Initialize wandb
        wandb_kwargs = {
            "project": project,
            "name": name,
            "config": config
        }
        
        if group:
            wandb_kwargs["group"] = group
            
        try:
            wandb.init(**wandb_kwargs)
            logger.info(f"Initialized wandb logging: project={project}, name={name}")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {str(e)}")
            self.use_wandb = False
                
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        # Log to wandb
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log to wandb: {str(e)}")
                
    def log_image(self, name: str, image, step: Optional[int] = None) -> None:
        """
        Log an image to wandb.
        
        Args:
            name: Image name
            image: Image data
            step: Optional step number
        """
        if self.use_wandb:
            try:
                wandb.log({name: wandb.Image(image)}, step=step)
            except Exception as e:
                logger.error(f"Failed to log image to wandb: {str(e)}")
                
    def log_model(self, model: torch.nn.Module) -> None:
        """
        Log model architecture to wandb.
        
        Args:
            model: PyTorch model
        """
        if self.use_wandb:
            try:
                wandb.watch(model)
            except Exception as e:
                logger.error(f"Failed to log model to wandb: {str(e)}")
                
    def close(self) -> None:
        """
        Close all logging connections.
        """
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error closing wandb: {str(e)}")