import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from typing import List, Tuple, Optional
import cv2
from PIL import Image


def denormalize_image(tensor: torch.Tensor, 
                     mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (List[float]): Mean values used for normalization
        std (List[float]): Std values used for normalization
    
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


def show_batch(dataloader, 
               class_names: List[str] = None,
               num_images: int = 8,
               figsize: Tuple[int, int] = (15, 8),
               save_path: str = None):
    """
    Display a batch of images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names (List[str]): Names of classes
        num_images (int): Number of images to display
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    # Get a batch of images
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Select subset of images
    num_images = min(num_images, len(images))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Create subplot
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(num_images):
        # Denormalize image
        img = denormalize_image(images[idx])
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()
        
        # Display image
        axes[idx].imshow(img_np)
        axes[idx].axis('off')
        
        # Set title
        if class_names:
            title = f'{class_names[labels[idx]]}'
        else:
            title = f'Label: {labels[idx]}'
        axes[idx].set_title(title)
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int],
                         class_names: List[str] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        class_names (List[str]): Names of classes
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def visualize_predictions(model,
                         dataloader,
                         device: str = 'cuda',
                         class_names: List[str] = None,
                         num_images: int = 8,
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: str = None):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device (str): Device to use
        class_names (List[str]): Names of classes
        num_images (int): Number of images to display
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    model.eval()
    
    # Get a batch of images
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Select subset
    num_images = min(num_images, len(images))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Get predictions
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        predicted = predicted.cpu()
        probabilities = probabilities.cpu()
    
    # Create subplot
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(num_images):
        # Denormalize image
        img = denormalize_image(images[idx])
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()
        
        # Display image
        axes[idx].imshow(img_np)
        axes[idx].axis('off')
        
        # Prepare title
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        confidence = probabilities[idx][pred_label].item()
        
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            title = f'True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}'
        else:
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}'
        
        # Color coding: green for correct, red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(title, color=color, fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels: List[int],
                           class_names: List[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: str = None):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        labels (List[int]): List of labels
        class_names (List[str]): Names of classes
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    
    if class_names:
        names = [class_names[i] for i in unique_labels]
        plt.bar(names, counts)
        plt.xticks(rotation=45, ha='right')
    else:
        plt.bar(unique_labels, counts)
        plt.xlabel('Class')
    
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def visualize_feature_maps(model,
                          image: torch.Tensor,
                          layer_name: str = None,
                          device: str = 'cuda',
                          max_maps: int = 16,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: str = None):
    """
    Visualize feature maps from a specific layer.
    
    Args:
        model: PyTorch model
        image (torch.Tensor): Input image tensor
        layer_name (str): Name of the layer to visualize
        device (str): Device to use
        max_maps (int): Maximum number of feature maps to show
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    model.eval()
    
    # Hook function to capture feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.cpu().detach())
    
    # Register hook
    if layer_name:
        for name, module in model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
    else:
        # Use first conv layer by default
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                handle = module.register_forward_hook(hook_fn)
                break
    
    # Forward pass
    with torch.no_grad():
        image_device = image.unsqueeze(0).to(device)
        _ = model(image_device)
    
    # Remove hook
    handle.remove()
    
    if not feature_maps:
        print("No feature maps captured. Check layer name.")
        return
    
    # Get feature maps
    maps = feature_maps[0].squeeze(0)  # Remove batch dimension
    num_maps = min(maps.shape[0], max_maps)
    
    # Create subplot
    cols = min(4, num_maps)
    rows = (num_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(num_maps):
        feature_map = maps[idx].numpy()
        
        axes[idx].imshow(feature_map, cmap='viridis')
        axes[idx].axis('off')
        axes[idx].set_title(f'Feature Map {idx}')
    
    # Hide unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Feature Maps from Layer: {layer_name or "Conv Layer"}')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps visualization saved to {save_path}")
    
    plt.show()


def save_sample_images(dataloader,
                      save_dir: str = 'data/samples',
                      num_samples: int = 10,
                      class_names: List[str] = None):
    """
    Save sample images from the dataset to disk.
    
    Args:
        dataloader: PyTorch DataLoader
        save_dir (str): Directory to save images
        num_samples (int): Number of samples to save per class
        class_names (List[str]): Names of classes
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Dictionary to track samples per class
    class_counts = {}
    saved_count = 0
    
    for images, labels in dataloader:
        for i, (image, label) in enumerate(zip(images, labels)):
            label = label.item()
            
            # Check if we need more samples for this class
            if class_counts.get(label, 0) < num_samples:
                # Denormalize image
                img = denormalize_image(image)
                img = torch.clamp(img, 0, 1)
                
                # Convert to PIL Image
                img_pil = transforms.ToPILImage()(img)
                
                # Create filename
                class_name = class_names[label] if class_names else f'class_{label}'
                filename = f'{class_name}_{class_counts.get(label, 0):03d}.png'
                filepath = os.path.join(save_dir, filename)
                
                # Save image
                img_pil.save(filepath)
                
                # Update counters
                class_counts[label] = class_counts.get(label, 0) + 1
                saved_count += 1
                
                print(f'Saved: {filepath}')
        
        # Check if we have enough samples
        if all(count >= num_samples for count in class_counts.values()):
            break
    
    print(f'Saved {saved_count} sample images to {save_dir}')
