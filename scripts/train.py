#!/usr/bin/env python3
"""
Training script for computer vision models.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import sys
import os
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from models.cnn_model import SimpleCNN, SimpleResNet
from data.dataset import get_dataloaders
from utils.training import Trainer, get_optimizer, get_scheduler


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config):
    """Create model based on configuration."""
    model_name = config['model']['name']
    
    if model_name == 'SimpleCNN':
        model = SimpleCNN(
            num_classes=config['model']['num_classes'],
            input_channels=config['model']['input_channels']
        )
    elif model_name == 'SimpleResNet':
        model = SimpleResNet(
            num_classes=config['model']['num_classes'],
            input_channels=config['model']['input_channels']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train computer vision model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['dataloader']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Model: {config['model']['name']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['dataloader']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        input_size=tuple(config['dataset']['input_size'])
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer'],
        lr=config['training']['learning_rate'],
        **config['training']['optimizer_params']
    )
    
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training']['scheduler']['name'],
        **{k: v for k, v in config['training']['scheduler'].items() if k != 'name'}
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Start training
    print("\nStarting training...")
    
    # Create save directories
    os.makedirs(config['logging']['model_save_dir'], exist_ok=True)
    os.makedirs(config['logging']['figure_save_dir'], exist_ok=True)
    
    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        save_best=config['logging']['save_best_model'],
        save_path=f"{config['logging']['model_save_dir']}/best_model.pth"
    )
    
    # Save final model
    if config['logging']['save_last_model']:
        final_model_path = f"{config['logging']['model_save_dir']}/final_model.pth"
        model.save_model(final_model_path)
    
    # Plot training history
    if config['visualization']['plot_training_history']:
        trainer.plot_training_history(
            save_path=f"{config['logging']['figure_save_dir']}/training_history.png"
        )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    results = trainer.evaluate(
        test_loader=val_loader,
        class_names=config['dataset']['class_names']
    )
    
    print(f"\nFinal Results:")
    print(f"Validation Loss: {results['test_loss']:.4f}")
    print(f"Validation Accuracy: {results['test_accuracy']:.4f}")
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
