import os
import logging
import time
from typing import Dict, Any, Optional
import torch
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logging_utils import MetricsLogger

logger = logging.getLogger(__name__)

class Trainer:
    """
    Enhanced trainer class with fixed metrics calculation and optimized logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        metrics_logger: MetricsLogger,
        device: torch.device,
        log_interval: int = 500,
        mixed_precision: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            metrics_logger: Metrics logging utility
            device: Device to use for training
            log_interval: How often to log metrics (in steps)
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_logger = metrics_logger
        self.device = device
        self.log_interval = log_interval
        self.mixed_precision = mixed_precision and device.type == 'cuda' and torch.cuda.is_available()
        
        # Initialize mixed precision scaler if needed
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision else None
        
        # Optimize CUDA operations
        if device.type == 'cuda':
            # Set higher benchmark mode for better performance with fixed input sizes
            torch.backends.cudnn.benchmark = True
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
            
            # Set higher memory allocation efficiency
            torch.backends.cuda.max_split_size_mb = 128
            
            logger.info("CUDA optimizations enabled for better GPU utilization")
        
        logger.info(f"Trainer initialized with device={device}, mixed_precision={self.mixed_precision}")
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        steps_per_epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            steps_per_epoch: Number of steps per epoch
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        start_time = time.time()
        
        # Use tqdm with minimal output and percentage-based updates
        progress_bar = tqdm(
            train_loader, 
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}",
            disable=False,
            mininterval=30.0,
            bar_format='{desc}: {percentage:3.0f}%|{bar:10}{r_bar}',
            position=0,
            leave=True,
        )
        
        last_percentage = -1  # Track last logged percentage
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            if batch_idx >= steps_per_epoch:
                break
                
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = labels.size(0)
            
            # Forward pass with mixed precision if enabled
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                self.scaler.scale(loss).backward() if self.scaler else loss.backward()
                self.self.scaler.step(self.optimizer); self.scaler.update() if self.scaler else self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            running_correct += batch_correct
            total_samples += batch_size
            
            # Calculate batch accuracy (properly bounded between 0 and 1)
            batch_accuracy = batch_correct / batch_size
            
            # Log batch metrics at specified intervals
            if (batch_idx + 1) % self.log_interval == 0:
                # Get both learning rates
                backbone_lr = self.scheduler.get_last_lr()[0]
                classifier_lr = self.scheduler.get_last_lr()[1]
                step = epoch * steps_per_epoch + batch_idx
                
                metrics = {
                    'train/batch_loss': loss.item(),
                    'train/batch_accuracy': batch_accuracy,
                    'charts/lr_backbone': backbone_lr,
                    'charts/lr_classifier': classifier_lr,
                    'charts/epoch': epoch + (batch_idx / steps_per_epoch)
                }
                
                self.metrics_logger.log(metrics, step=step)
            
            # Only update progress bar at percentage point intervals
            current_percentage = int(100 * batch_idx / steps_per_epoch)
            if current_percentage > last_percentage:
                last_percentage = current_percentage
                # Get both learning rates for display
                backbone_lr = self.scheduler.get_last_lr()[0]
                classifier_lr = self.scheduler.get_last_lr()[1]
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_accuracy:.4f}',
                    'b_lr': f'{backbone_lr:.6f}',
                    'c_lr': f'{classifier_lr:.6f}'
                })
                progress_bar.update(0)  # Force refresh
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_accuracy = running_correct / total_samples
        epoch_time = time.time() - start_time
        
        metrics = {
            'train/loss': epoch_loss,
            'train/accuracy': epoch_accuracy,
            'train/epoch_time': epoch_time,
            'charts/epoch': epoch
        }
        
        self.metrics_logger.log(metrics)
        
        logger.info(
            f"Epoch {epoch+1} - "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        return metrics
        
    def evaluate(
        self,
        test_loader: DataLoader,
        epoch: int,
        test_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            epoch: Current epoch number
            test_steps: Number of steps to evaluate (if None, use all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total_samples = 0
        start_time = time.time()
        
        # Use tqdm with minimal output and percentage-based updates
        progress_bar = tqdm(
            test_loader,
            total=test_steps,
            desc="Evaluating",
            disable=False,
            mininterval=30.0,
            bar_format='{desc}: {percentage:3.0f}%|{bar:10}{r_bar}',
            position=0,
            leave=True,
        )
        
        last_percentage = -1  # Track last logged percentage
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                if test_steps is not None and batch_idx >= test_steps:
                    break
                    
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = labels.size(0)
                
                # Forward pass with mixed precision if enabled
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = F.cross_entropy(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                
                # Update metrics
                test_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                correct += batch_correct
                total_samples += batch_size
                
                # Only update progress bar at percentage point intervals
                if test_steps:
                    current_percentage = int(100 * batch_idx / test_steps)
                    if current_percentage > last_percentage:
                        last_percentage = current_percentage
                        batch_accuracy = batch_correct / batch_size
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{batch_accuracy:.4f}'
                        })
                        progress_bar.update(0)  # Force refresh
        
        # Calculate evaluation metrics
        test_loss /= total_samples
        test_accuracy = correct / total_samples
        eval_time = time.time() - start_time
        
        metrics = {
            'test/loss': test_loss,
            'test/accuracy': test_accuracy,
            'test/epoch_time': eval_time,
            'charts/epoch': epoch 
        }
        
        self.metrics_logger.log(metrics)
        
        logger.info(
            f"Evaluation - "
            f"Loss: {test_loss:.4f}, "
            f"Accuracy: {test_accuracy:.4f}, "
            f"Time: {eval_time:.2f}s"
        )
        
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        steps_per_epoch: int,
        test_steps: Optional[int] = None,
        save_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 5
    ) -> Dict[str, float]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of epochs to train
            steps_per_epoch: Number of steps per epoch
            test_steps: Number of steps to evaluate (if None, use all)
            save_path: Path to save best model (if None, don't save)
            checkpoint_dir: Directory to save periodic checkpoints (if None, don't save)
            checkpoint_frequency: How often to save checkpoints (in epochs)
            
        Returns:
            Dictionary of final metrics
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        best_accuracy = 0.0
        best_epoch = -1
        
        # Create checkpoint directory if needed
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Will save checkpoints every {checkpoint_frequency} epochs to {checkpoint_dir}")
        
        for epoch in range(epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch, steps_per_epoch)
            
            # Evaluate
            eval_metrics = self.evaluate(test_loader, epoch, test_steps)
            
            # Save best model if requested
            if save_path is not None and eval_metrics['test/accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['test/accuracy']
                best_epoch = epoch
                
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'accuracy': best_accuracy
                    }, save_path)
                    
                    logger.info(f"Saved best model at epoch {epoch+1} with accuracy {best_accuracy:.4f}")
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")
            
            # Save periodic checkpoint if requested
            if checkpoint_dir is not None and (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_metrics': train_metrics,
                        'eval_metrics': eval_metrics
                    }, checkpoint_path)
                    
                    logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {str(e)}")
        
        logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f} at epoch {best_epoch+1}")
        
        return {
            'best_accuracy': best_accuracy,
            'best_epoch': best_epoch
        }


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    steps_per_epoch: int,
    test_steps: Optional[int] = None,
    device: torch.device = None,
    log_interval: int = 500,
    save_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_frequency: int = 5,
    mixed_precision: bool = True,
    project_name: str = "deepdraw",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Train a model with enhanced logging and metrics.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        epochs: Number of epochs to train
        train_loader: Training data loader
        test_loader: Test data loader
        steps_per_epoch: Number of steps per epoch
        test_steps: Number of steps to evaluate (if None, use all)
        device: Device to use for training (if None, use auto-detection)
        log_interval: How often to log metrics (in steps)
        save_path: Path to save best model (if None, don't save)
        checkpoint_dir: Directory to save periodic checkpoints (if None, don't save)
        checkpoint_frequency: How often to save checkpoints (in epochs)
        mixed_precision: Whether to use mixed precision training
        project_name: Project name for wandb
        run_name: Run name for wandb (if None, use auto-generated)
        config: Configuration dictionary for wandb
        
    Returns:
        Dictionary of final metrics
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Move model to device
    model = model.to(device)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        project=project_name,
        name=run_name or "training_run",
        config=config or {}
    )
    
    # Log model architecture
    metrics_logger.log_model(model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_logger=metrics_logger,
        device=device,
        log_interval=log_interval,
        mixed_precision=mixed_precision
    )
    
    # Train model
    results = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        test_steps=test_steps,
        save_path=save_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=checkpoint_frequency
    )
    
    # Close metrics logger
    metrics_logger.close()
    
    return results