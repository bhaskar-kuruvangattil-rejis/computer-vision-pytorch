import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Base class for all PyTorch models.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        pass
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        """
        Save the model state dict.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path, device='cpu'):
        """
        Load the model state dict.
        
        Args:
            path (str): Path to the saved model
            device (str): Device to load the model on
        """
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
