"""
Training Utilities for Advanced LSTM with Temporal Attention
===========================================================

This module provides training, evaluation, and visualization utilities
for the AdvancedLSTMTemporal model, including mixed precision training,
adversarial training loops, and performance monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import warnings


class IoTTimeSeriesDataset(Dataset):
    """
    Dataset class for IoT time series data with sliding window approach.
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 seq_length: int = 100, stride: int = 10,
                 augment: bool = True):
        self.data = data
        self.labels = labels
        self.seq_length = seq_length
        self.stride = stride
        self.augment = augment
        
        # Create sequences
        self.sequences = []
        self.sequence_labels = []
        
        for i in range(0, len(data) - seq_length, stride):
            self.sequences.append(data[i:i + seq_length])
            # Use majority voting for sequence label
            self.sequence_labels.append(
                1 if np.mean(labels[i:i + seq_length]) > 0.5 else 0
            )
        
        self.sequences = np.array(self.sequences)
        self.sequence_labels = np.array(self.sequence_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        label = self.sequence_labels[idx]
        
        # Data augmentation for IoT time series
        if self.augment and np.random.random() < 0.5:
            # Add Gaussian noise
            noise_level = np.random.uniform(0.01, 0.05)
            seq += np.random.normal(0, noise_level, seq.shape)
            
            # Random scaling
            scale_factor = np.random.uniform(0.9, 1.1)
            seq *= scale_factor
            
            # Temporal jittering
            if np.random.random() < 0.3:
                shift = np.random.randint(-2, 3)
                seq = np.roll(seq, shift, axis=0)
        
        return torch.FloatTensor(seq), torch.LongTensor([label]).squeeze()


class AdversarialTrainer:
    """
    Trainer class with adversarial training support.
    """
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-3, 
                 adversarial_weight: float = 0.1):
        self.model = model
        self.device = device
        self.adversarial_weight = adversarial_weight
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                          weight_decay=1e-5)
        self.discriminator_optimizer = None
        if model.use_adversarial:
            self.discriminator_optimizer = torch.optim.AdamW(
                model.discriminator.parameters(), lr=learning_rate * 2
            )
        
        # Mixed precision scaler
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(device_type)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with adversarial training.
        """
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        adversarial_losses = []
        uncertainty_losses = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            if self.discriminator_optimizer:
                self.discriminator_optimizer.zero_grad()
            
            # Forward pass with mixed precision
            device_type = self.device.type
            with autocast(device_type):
                outputs = self.model(data, return_attention=True, 
                                   adversarial_training=True)
                logits = outputs['logits']
                
                # Classification loss
                classification_loss = F.cross_entropy(logits, labels)
                
                # Uncertainty loss (KL divergence)
                uncertainty_loss = 0
                if 'uncertainty' in outputs and self.model.training:
                    uncertainty_loss = outputs['uncertainty'].mean() * 0.01
                    uncertainty_losses.append(uncertainty_loss.item())
                
                # Adversarial loss
                adversarial_loss = 0
                if self.model.use_adversarial and 'discriminator_output' in outputs:
                    # Create fake labels (all ones for generated features)
                    fake_labels = torch.ones_like(outputs['discriminator_output'])
                    adversarial_loss = F.binary_cross_entropy(
                        outputs['discriminator_output'], fake_labels
                    ) * self.adversarial_weight
                    adversarial_losses.append(adversarial_loss.item())
                
                # Total loss
                total_batch_loss = classification_loss + uncertainty_loss + adversarial_loss
            
            # Backward pass with gradient scaling
            self.scaler.scale(total_batch_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer steps
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update discriminator separately if using adversarial training
            if self.model.use_adversarial and self.discriminator_optimizer:
                self._update_discriminator(data, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += total_batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_batch_loss.item(),
                'acc': total_correct / total_samples,
                'adv_loss': np.mean(adversarial_losses[-10:]) if adversarial_losses else 0
            })
        
        # Learning rate scheduling
        self.scheduler.step()
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = total_correct / total_samples
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'adversarial_loss': np.mean(adversarial_losses) if adversarial_losses else 0,
            'uncertainty_loss': np.mean(uncertainty_losses) if uncertainty_losses else 0
        }
    
    def _update_discriminator(self, data: torch.Tensor, labels: torch.Tensor):
        """
        Update discriminator for adversarial training.
        """
        self.model.eval()
        self.model.discriminator.train()
        
        with torch.no_grad():
            outputs = self.model(data)
            features = self.model.memory_network(
                self.model.input_projection(data).mean(dim=1)
            )
        
        # Real samples (benign traffic)
        real_features = features[labels == 0]
        fake_features = features[labels == 1]
        
        if len(real_features) > 0 and len(fake_features) > 0:
            # Discriminator loss
            real_pred = self.model.discriminator(real_features.detach())
            fake_pred = self.model.discriminator(fake_features.detach())
            
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            
            disc_loss = (real_loss + fake_loss) / 2
            
            disc_loss.backward()
            self.discriminator_optimizer.step()
        
        self.model.train()
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        """
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        for data, labels in tqdm(val_loader, desc='Validation'):
            data, labels = data.to(self.device), labels.to(self.device)
            
            device_type = self.device.type
            with autocast(device_type):
                outputs = self.model(data)
                logits = outputs['logits']
                loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if 'uncertainty' in outputs:
                all_uncertainties.extend(outputs['uncertainty'][:, 1].cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        val_accuracy = total_correct / total_samples
        auc_score = roc_auc_score(all_labels, all_predictions)
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        metrics = {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'auc': auc_score
        }
        
        if all_uncertainties:
            metrics['mean_uncertainty'] = np.mean(all_uncertainties)
            metrics['uncertainty_std'] = np.std(all_uncertainties)
        
        return metrics


def visualize_attention_weights(model: nn.Module, data_sample: torch.Tensor,
                               save_path: Optional[str] = None):
    """
    Visualize attention weights for interpretability.
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(data_sample.unsqueeze(0), return_attention=True)
        
        if 'attention_weights' in outputs:
            attention = outputs['attention_weights'].squeeze().cpu().numpy()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(attention.mean(axis=0), cmap='Blues', cbar=True)
            plt.title('Temporal Attention Weights')
            plt.xlabel('Time Steps')
            plt.ylabel('Time Steps')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()


def plot_training_history(trainer: AdversarialTrainer, save_path: Optional[str] = None):
    """
    Plot training history including losses and accuracies.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(epochs, trainer.train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, trainer.val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate schedule
    lrs = [trainer.scheduler.get_last_lr()[0] for _ in epochs]
    ax3.plot(epochs, lrs, 'g-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True)
    ax3.set_yscale('log')
    
    # Loss components
    ax4.text(0.1, 0.5, 'Loss Components Analysis\n\n' +
             f'Final Training Loss: {trainer.train_losses[-1]:.4f}\n' +
             f'Final Validation Loss: {trainer.val_losses[-1]:.4f}\n' +
             f'Final Training Accuracy: {trainer.train_accuracies[-1]:.4f}\n' +
             f'Final Validation Accuracy: {trainer.val_accuracies[-1]:.4f}',
             transform=ax4.transAxes, fontsize=12, verticalalignment='center')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def analyze_model_predictions(model: nn.Module, test_loader: DataLoader,
                            device: torch.device, save_dir: Optional[str] = None):
    """
    Comprehensive analysis of model predictions including uncertainty.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Analyzing predictions'):
            data = data.to(device)
            outputs = model(data)
            
            probs = F.softmax(outputs['logits'], dim=1)
            all_predictions.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if 'uncertainty' in outputs:
                all_uncertainties.extend(outputs['uncertainty'][:, 1].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Receiver Operating Characteristic')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    axes[0, 1].plot(recall, precision, color='green', lw=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions > 0.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Uncertainty distribution
    if all_uncertainties:
        all_uncertainties = np.array(all_uncertainties)
        axes[1, 1].hist(all_uncertainties[all_labels == 0], bins=50, alpha=0.5,
                       label='Benign', color='green')
        axes[1, 1].hist(all_uncertainties[all_labels == 1], bins=50, alpha=0.5,
                       label='Attack', color='red')
        axes[1, 1].set_xlabel('Uncertainty')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Uncertainty Distribution')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No uncertainty estimates available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/prediction_analysis.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    return {
        'auc': roc_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'uncertainties': all_uncertainties if all_uncertainties else None
    }


if __name__ == "__main__":
    print("Training utilities loaded successfully!")
    print("This module provides:")
    print("- IoTTimeSeriesDataset: Dataset class for time series")
    print("- AdversarialTrainer: Advanced training with adversarial components")
    print("- Visualization functions for attention and training history")
    print("- Comprehensive model analysis tools")