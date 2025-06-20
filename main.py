#!/usr/bin/env python
import os
import argparse
import logging
import torch
from data.dataloader import get_loader, count_samples
from models.build_model import build_model
from trainers.trainer import train_model
from utils.utils import seed_all, load_config, validate_config, setup_gpu_environment, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepDraw Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--base_config', type=str, default=None, help='Optional path to base config file')
    parser.add_argument('--name_appendix', type=str, default='', help='Name appendix for logging')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval in steps')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--checkpoint_frequency', type=int, default=5, help='Save checkpoints every N epochs')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    return parser.parse_args()


def main():
    """Main entry point for training."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Load and validate configuration
        config = load_config(args.config, args.base_config)
        validate_config(config)
        
        # Set random seed for reproducibility
        seed_all(config.get('seed', 42))
        
        # Setup device and GPU environment
        device = get_device()
        setup_gpu_environment()
        
        # Create save directory if needed
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"best_model{args.name_appendix}.pt")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(save_dir, f"checkpoints{args.name_appendix}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load data with optimized settings for better GPU utilization
        logger.info("Loading datasets...")
        train_loader = get_loader(
            config['data']['train_dir'], 
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=args.num_workers,
            prefetch_factor=2,  # Prefetch batches
            pin_memory=True     # Faster data transfer to GPU
        )
        
        # Use larger batch size for evaluation to improve GPU utilization
        eval_batch_size = config['train']['batch_size'] * 2  # Double the batch size for evaluation
        test_loader = get_loader(
            config['data']['test_dir'], 
            batch_size=eval_batch_size,
            num_workers=args.num_workers,
            prefetch_factor=2,
            pin_memory=True
        )
        
        # Count samples for steps calculation
        steps_per_epoch = count_samples(config['data']['train_dir']) // config['train']['batch_size']
        test_steps = count_samples(config['data']['test_dir']) // eval_batch_size
        
        logger.info(f"Training steps per epoch: {steps_per_epoch}")
        logger.info(f"Evaluation steps: {test_steps}")

        # Build model
        logger.info(f"Building model: {config['model']['type']}")
        model = build_model(
            backbone_name=config['model']['type'],
            num_classes=config['model']['num_classes'],
            freeze_backbone=config['model'].get('freeze_backbone', False),
            freeze_blocks=config['model'].get('freeze_blocks', None)
        ).to(device)

        # Separate backbone and classifier parameters for different learning rates
        backbone_params, classifier_params = [], []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

        # Create optimizer
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': config['train']['backbone_lr']},
            {'params': classifier_params, 'lr': config['train']['classifier_lr']}
        ], weight_decay=config['train']['weight_decay'])

        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config['train']['backbone_lr'], config['train']['classifier_lr']],
            steps_per_epoch=steps_per_epoch,
            epochs=config['train']['epochs'],
            pct_start=0.3
        )

        # Setup wandb configuration
        project = config.get('wandb', {}).get('project', 'deepdraw')
        model_name = config['model']['type']
        optimizer_name = 'AdamW'
        scheduler_name = 'OneCycleLR'
        lr = config['train']['classifier_lr']
        appendix = args.name_appendix

        run_name = f"{model_name}_{optimizer_name}_{scheduler_name}_{lr}"
        if appendix:
            run_name += f"_{appendix}"

        # Train model
        logger.info(f"Starting training for {config['train']['epochs']} epochs")
        results = train_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config['train']['epochs'],
            train_loader=train_loader,
            test_loader=test_loader,
            steps_per_epoch=steps_per_epoch,
            test_steps=test_steps,
            device=device,
            log_interval=args.log_interval,
            save_path=model_save_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=args.checkpoint_frequency,
            mixed_precision=args.mixed_precision,
            project_name=project,
            run_name=run_name,
            config=config
        )
        
        logger.info(f"Training completed. Best accuracy: {results['best_accuracy']:.4f} at epoch {results['best_epoch']+1}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
