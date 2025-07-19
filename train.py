import os
import torch
import yaml
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import warnings
from tqdm import tqdm

# Import actual model components
from models.backbones.dinov2 import load_dinov2
from models.classifiers.dino_classifier import DinoClassifier

# Import the provided dataloader
from dataloader import get_loader, load_categories  # Assuming dataloader.py is copied as data.py

# Dummy utils.py for now, as load_dinov2 depends on it
# In a real scenario, you'd have a proper utils.py with these functions
class DummyLogger:
    def info(self, message): print(f"INFO: {message}")
    def warning(self, message): print(f"WARNING: {message}")
    def error(self, message): print(f"ERROR: {message}")
    def isEnabledFor(self, level): return True  # Always enable for now
    def debug(self, message): print(f"DEBUG: {message}")

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Mocking utils.utils for load_dinov2 to work
# In a real setup, you would have these in your utils/utils.py
class MockUtils:
    @staticmethod
    def count_trainable_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

    @staticmethod
    def get_block_parameter_count(model):
        # This is a simplified mock. Actual implementation would traverse model structure.
        return []

import sys
sys.modules["utils.utils"] = MockUtils
sys.modules["utils"] = MockUtils  # For logger import


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Suppress specific UserWarnings from dinov2.layers
    warnings.filterwarnings("ignore", category=UserWarning, module="dinov2.layers")

    # Initialize wandb
    wandb.init(project='DinoV2-Zeroshot', job_type='zeroshot_prediction')

    # set wandb run name
    wandb.run.name = 'DinoV2_finetuned_epoch_2'

    # Configuration (can be loaded from a YAML file or defined here)
    config = {
        'zeroshot_dir': '/hpcwork/lect0149/quickdraw_zeroshot',
        'categories_file': 'categories_zeroshot.txt',
        'batch_size': 64,
        'num_workers': 4,
        'checkpoint_path': '/home/lect0149/gitlab/outputs/checkpoints_/checkpoint_epoch_2.pt',
        'model_name': 'dinov2_vitl14_reg',
        'backbone_output_dim': 1024
    }
    
    # Update config with wandb config for logging
    wandb.config.update(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set environment variable to disable xformers if running on CPU
    if device.type == 'cpu':
        os.environ["DINOV2_DISABLE_XFORMERS"] = "1"
        print("DINOV2_DISABLE_XFORMERS set to 1 for CPU execution.")

    # Load zeroshot categories
    categories = load_categories(config['categories_file'])
    num_classes = len(categories)
    print(f"Number of zeroshot classes: {num_classes}")

    # Load zeroshot dataloader using the provided get_loader function
    dataloader = get_loader(
        tar_dir=config['zeroshot_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        categories_file=config['categories_file'],
        image_size=224,
        normalize=True,
        pin_memory=torch.cuda.is_available()
    )

    # Load backbone and classifier
    backbone, feature_dim = load_dinov2(freeze_blocks=None)  # Assuming no freezing for inference
    
    # Initialize a temporary classifier with the original number of training classes (275)
    # to load the checkpoint, then replace the final layer.
    temp_classifier = DinoClassifier(backbone=backbone, feature_dim=feature_dim, num_classes=275)  # Assuming 275 was the original training classes

    # Load checkpoint
    try:
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        # Load state_dict, but ignore the final layer if it doesn't match
        model_state_dict = checkpoint['model_state_dict']
        
        # Filter out the last layer's weights if it doesn't match
        # This assumes the last layer is named 'classifier.8.weight' and 'classifier.8.bias'
        # based on the previous error message and the structure of DinoClassifier
        # If your DinoClassifier has a different structure, this needs adjustment.
        pretrained_dict = {k: v for k, v in model_state_dict.items() if not k.startswith('classifier.8.')}
        
        # Load the filtered state_dict into the temporary classifier
        temp_classifier.load_state_dict(pretrained_dict, strict=False)
        print(f"Model loaded from {config['checkpoint_path']} (excluding final layer).")

        # Now, create the actual classifier with 70 classes and copy over the loaded weights
        classifier = DinoClassifier(backbone=backbone, feature_dim=feature_dim, num_classes=num_classes)
        
        # Copy parameters from temp_classifier to classifier, excluding the final layer
        for name, param in temp_classifier.named_parameters():
            if not name.startswith('classifier.8.') and name in classifier.state_dict() and classifier.state_dict()[name].shape == param.shape:
                classifier.state_dict()[name].copy_(param)
            else:
                print(f"Warning: Skipping parameter {name} due to shape mismatch or absence in new classifier.")
        
        print("Final classification layer re-initialized for 70 classes.")

    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {config['checkpoint_path']}. Please provide a valid path.")
        return  # Exit if checkpoint is not found
    except KeyError:
        print(f"Error: 'model_state_dict' key not found in checkpoint. Please check checkpoint structure.")
        return
    except Exception as e:
        print(f"An error occurred during checkpoint loading: {e}")
        return

    classifier.to(device)
    classifier.eval()  # Set model to evaluation mode

    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    print("Starting zeroshot prediction...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Processing batches")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Calculate top-k accuracy
            _, predicted = outputs.topk(10, 1, True, True)
            predicted = predicted.t()
            
            correct = predicted.eq(labels.view(1, -1).expand_as(predicted))

            correct_1 += correct[:1].reshape(-1).float().sum(0).item()
            correct_5 += correct[:5].reshape(-1).float().sum(0).item()
            correct_10 += correct[:10].reshape(-1).float().sum(0).item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    top1_acc = correct_1 / total_samples
    top5_acc = correct_5 / total_samples
    top10_acc = correct_10 / total_samples

    print(f"\nZeroshot Prediction Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Top-10 Accuracy: {top10_acc:.4f}")
    print(f"Random Guessing (1/{num_classes}): {1/num_classes:.4f}")

    # Log results to wandb
    wandb.log({
        'zeroshot/average_loss': avg_loss,
        'zeroshot/top1_accuracy': top1_acc,
        'zeroshot/top5_accuracy': top5_acc,
        'zeroshot/top10_accuracy': top10_acc,
        'zeroshot/random_guessing_accuracy': 1/num_classes
    })

    wandb.finish()

if __name__ == '__main__':
    main()