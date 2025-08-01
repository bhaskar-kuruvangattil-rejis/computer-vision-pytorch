import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import time


class Trainer:
    """
    A comprehensive trainer class for PyTorch models.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'auto',
                 criterion: nn.Module = None,
                 optimizer: optim.Optimizer = None,
                 scheduler: optim.lr_scheduler._LRScheduler = None):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train
            device (str): Device to use ('cuda', 'cpu', or 'auto')
            criterion (nn.Module): Loss function
            optimizer (optim.Optimizer): Optimizer
            scheduler: Learning rate scheduler
        """
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # Default criterion and optimizer if not provided
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {model.count_parameters():,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                avg_loss = total_loss / len(pbar)
                acc = 100. * correct / total
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader = None, 
              epochs: int = 10,
              save_best: bool = True,
              save_path: str = 'results/models/best_model.pth') -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs
            save_best (bool): Whether to save the best model
            save_path (str): Path to save the best model
            
        Returns:
            dict: Training history
        """
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save best model
                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.model.save_model(save_path)
                    print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            else:
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f} seconds')
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader, class_names: List[str] = None) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            class_names (List[str]): Names of classes for classification report
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_loss = total_loss / len(test_loader)
        
        print(f'Test Loss: {avg_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Classification report
        if class_names:
            print('\nClassification Report:')
            print(classification_report(all_targets, all_predictions, 
                                      target_names=class_names))
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def plot_training_history(self, save_path: str = 'results/figures/training_history.png'):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.history['train_loss']:
            print("No training history to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        if self.history['val_loss']:
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        if self.history['val_acc']:
            ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved to {save_path}")


def get_optimizer(model: nn.Module, 
                 optimizer_name: str = 'adam', 
                 lr: float = 0.001, 
                 **kwargs) -> optim.Optimizer:
    """
    Get optimizer by name.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer
        lr (float): Learning rate
        **kwargs: Additional optimizer parameters
        
    Returns:
        optim.Optimizer: Optimizer
    """
    
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adamw': optim.AdamW
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not supported. "
                        f"Supported: {list(optimizers.keys())}")
    
    optimizer_class = optimizers[optimizer_name.lower()]
    return optimizer_class(model.parameters(), lr=lr, **kwargs)


def get_scheduler(optimizer: optim.Optimizer, 
                 scheduler_name: str = 'step', 
                 **kwargs) -> optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler by name.
    
    Args:
        optimizer (optim.Optimizer): Optimizer
        scheduler_name (str): Name of scheduler
        **kwargs: Additional scheduler parameters
        
    Returns:
        lr_scheduler: Learning rate scheduler
    """
    
    schedulers = {
        'step': optim.lr_scheduler.StepLR,
        'cosine': optim.lr_scheduler.CosineAnnealingLR,
        'exponential': optim.lr_scheduler.ExponentialLR,
        'plateau': optim.lr_scheduler.ReduceLROnPlateau
    }
    
    if scheduler_name.lower() not in schedulers:
        raise ValueError(f"Scheduler {scheduler_name} not supported. "
                        f"Supported: {list(schedulers.keys())}")
    
    scheduler_class = schedulers[scheduler_name.lower()]
    return scheduler_class(optimizer, **kwargs)
