"""
Advanced LSTM Temporal Attention - Complete Example
===================================================

This script demonstrates the complete usage of the AdvancedLSTMTemporal model
for IoT botnet detection, including data loading, training, evaluation, and
analysis with all advanced features.

Features demonstrated:
- Data preprocessing for N-BaIoT dataset
- Model training with adversarial and uncertainty components
- Mixed precision training
- Online learning adaptation
- Attention visualization and explainability
- Performance analysis and uncertainty quantification
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime
from typing import Optional, Tuple

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_lstm_temporal import AdvancedLSTMTemporal, create_model
from training_utils import IoTTimeSeriesDataset, AdversarialTrainer, visualize_attention_weights, plot_training_history, analyze_model_predictions

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8')


class NBaIoTDataProcessor:
    """
    Data processor for N-BaIoT dataset with advanced preprocessing.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_and_preprocess_data(self, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess N-BaIoT dataset.
        """
        print("Loading N-BaIoT dataset...")
        
        data_files = []
        labels = []
        
        # Find all CSV files in the data directory
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.csv') and not file.startswith('features'):
                    file_path = os.path.join(root, file)
                    
                    # Determine label from filename
                    if 'benign' in file.lower():
                        label = 'benign'
                    elif any(attack in file.lower() for attack in ['mirai', 'gafgyt']):
                        if 'mirai' in file.lower():
                            label = 'mirai'
                        else:
                            label = 'gafgyt'
                    else:
                        continue
                    
                    data_files.append((file_path, label))
        
        print(f"Found {len(data_files)} data files")
        
        # Load and combine data
        all_data = []
        all_labels = []
        
        for file_path, label in tqdm(data_files, desc="Loading files"):
            try:
                df = pd.read_csv(file_path)
                
                # Skip empty files
                if df.empty:
                    continue
                
                # Store feature names from first file
                if self.feature_names is None:
                    self.feature_names = df.columns.tolist()
                
                # Sample data if specified
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                
                all_data.append(df.values)
                all_labels.extend([label] * len(df))
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data files loaded successfully")
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.array(all_labels)
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Label distribution: {np.unique(y, return_counts=True)}")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Encode labels (binary: benign=0, attack=1)
        y_binary = np.where(y == 'benign', 0, 1)
        
        return X, y_binary
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        seq_length: int = 100, stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data.
        """
        print(f"Creating sequences with length={seq_length}, stride={stride}")
        
        sequences = []
        labels = []
        
        for i in tqdm(range(0, len(X) - seq_length, stride), desc="Creating sequences"):
            seq = X[i:i + seq_length]
            # Use majority voting for sequence label
            seq_label = 1 if np.mean(y[i:i + seq_length]) > 0.5 else 0
            
            sequences.append(seq)
            labels.append(seq_label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Label distribution: {np.unique(labels, return_counts=True)}")
        
        return sequences, labels


def train_advanced_lstm_model(data_path: str, config: Dict, args):
    """
    Complete training pipeline for the advanced LSTM model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"advanced_lstm_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    processor = NBaIoTDataProcessor(data_path)
    X, y = processor.load_and_preprocess_data(sample_size=args.sample_size)
    
    # Create sequences
    X_seq, y_seq = processor.create_sequences(
        X, y, seq_length=args.seq_length, stride=args.stride
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Create datasets
    train_dataset = IoTTimeSeriesDataset(X_train, y_train, augment=True)
    val_dataset = IoTTimeSeriesDataset(X_val, y_val, augment=False)
    test_dataset = IoTTimeSeriesDataset(X_test, y_test, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    config['input_size'] = X.shape[1]
    model = create_model(config).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AdversarialTrainer(model, device, learning_rate=args.learning_rate)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Adv Loss: {train_metrics['adversarial_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), 
                      f"{results_dir}/best_advanced_lstm_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break
        
        # Online adaptation demonstration
        if epoch % 5 == 0 and epoch > 0:
            print("Demonstrating online learning...")
            sample_batch = next(iter(train_loader))
            model.online_update(sample_batch[0][:5], sample_batch[1][:5])
        
        # Adapt architecture based on performance
        model.adapt_architecture(val_metrics)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f"{results_dir}/best_advanced_lstm_model.pth"))
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    
    if 'mean_uncertainty' in test_metrics:
        print(f"  Mean Uncertainty: {test_metrics['mean_uncertainty']:.4f}")
        print(f"  Uncertainty Std: {test_metrics['uncertainty_std']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Training history
    plot_training_history(trainer, f"{results_dir}/training_history.png")
    
    # Attention weights
    sample_data = torch.FloatTensor(X_test[0]).to(device)
    visualize_attention_weights(model, sample_data, 
                               f"{results_dir}/attention_visualization.png")
    
    # Comprehensive analysis
    analysis_results = analyze_model_predictions(model, test_loader, device, results_dir)
    
    # Attention explanations
    explanations = model.get_attention_explanations(top_k=10)
    if explanations:
        print("\nTop attention patterns:")
        for i, (score, idx) in enumerate(zip(explanations['top_attention_scores'], 
                                            explanations['top_attention_indices'])):
            print(f"  {i+1}. Position {idx}: {score:.4f}")
    
    # Save configuration and results
    import json
    results_summary = {
        'config': config,
        'args': vars(args),
        'best_val_auc': best_val_auc,
        'final_test_metrics': test_metrics,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_time': f"{trainer.scheduler.last_epoch} epochs"
    }
    
    with open(f"{results_dir}/experiment_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_dir}")
    print("Experiment completed successfully!")
    
    return model, trainer, results_summary


def main():
    parser = argparse.ArgumentParser(description='Advanced LSTM for IoT Time Series')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default=r'C:\Users\prpatel.CSE-LAB\Documents\IOT\IOT Research\IOT\data\N-BaLot',
                       help='Path to N-BaIoT dataset')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size per file (None for all data)')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='Sequence length for time series')
    parser.add_argument('--stride', type=int, default=10,
                       help='Stride for sequence creation')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of LSTM layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--memory_size', type=int, default=128,
                       help='Memory bank size')
    
    # Feature flags
    parser.add_argument('--no_adversarial', action='store_true',
                       help='Disable adversarial training')
    parser.add_argument('--no_uncertainty', action='store_true',
                       help='Disable uncertainty estimation')
    parser.add_argument('--no_adaptive', action='store_true',
                       help='Disable adaptive architecture')
    
    args = parser.parse_args()
    
    # Model configuration
    config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_classes': 2,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'memory_size': args.memory_size,
        'use_adversarial': not args.no_adversarial,
        'use_uncertainty': not args.no_uncertainty,
        'adaptive_architecture': not args.no_adaptive
    }
    
    print("Advanced LSTM with Temporal Attention - IoT Time Series")
    print("=" * 60)
    print(f"Configuration: {config}")
    print(f"Arguments: {vars(args)}")
    print("=" * 60)
    
    # Check data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return
    
    # Run training
    try:
        model, trainer, results = train_advanced_lstm_model(args.data_path, config, args)
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()