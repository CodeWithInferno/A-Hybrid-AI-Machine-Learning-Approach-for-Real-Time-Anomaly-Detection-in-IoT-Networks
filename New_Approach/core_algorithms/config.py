"""
Configuration file for Advanced LSTM with Temporal Attention
============================================================

This module contains all configuration parameters, hyperparameters,
and settings for the advanced LSTM model and training pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Core architecture
    input_size: int = 115  # N-BaIoT feature count
    hidden_size: int = 256
    num_layers: int = 3
    num_classes: int = 2  # Binary classification
    
    # Attention mechanism
    num_heads: int = 8
    attention_dropout: float = 0.1
    use_temporal_bias: bool = True
    attention_temperature: float = 1.0
    
    # Hierarchical LSTM
    hierarchical_levels: int = 3
    use_temporal_gates: bool = True
    
    # Memory augmentation
    memory_size: int = 128
    memory_dim: int = 64
    
    # Advanced features
    use_adversarial: bool = True
    use_uncertainty: bool = True
    adaptive_architecture: bool = True
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Optimizer settings
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = 'cosine_annealing'  # 'cosine_annealing', 'step', 'exponential'
    scheduler_params: Dict = None
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Adversarial training
    adversarial_weight: float = 0.1
    gradient_reversal_lambda: float = 0.1
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    val_frequency: int = 1  # Validate every N epochs
    
    def __post_init__(self):
        if self.scheduler_params is None:
            if self.scheduler_type == 'cosine_annealing':
                self.scheduler_params = {'T_0': 10, 'T_mult': 2}
            elif self.scheduler_type == 'step':
                self.scheduler_params = {'step_size': 10, 'gamma': 0.5}
            elif self.scheduler_type == 'exponential':
                self.scheduler_params = {'gamma': 0.95}


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Dataset paths
    data_path: str = r'C:\Users\prpatel.CSE-LAB\Documents\IOT\IOT Research\IOT\data\N-BaLot'
    
    # Sequence parameters
    seq_length: int = 100
    stride: int = 10
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Preprocessing
    normalize_features: bool = True
    handle_missing: str = 'zero'  # 'zero', 'mean', 'median', 'drop'
    
    # Data augmentation
    augment_training: bool = True
    noise_level: float = 0.05
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    temporal_jitter_prob: float = 0.3
    temporal_jitter_range: int = 2
    
    # Sampling
    sample_size_per_file: Optional[int] = None
    balance_classes: bool = True
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """Experiment and logging configuration."""
    
    # Experiment tracking
    experiment_name: str = "advanced_lstm_temporal"
    use_wandb: bool = False
    wandb_project: str = "iot-botnet-detection"
    
    # Logging
    log_frequency: int = 100  # Log every N batches
    save_frequency: int = 5   # Save checkpoint every N epochs
    
    # Visualization
    plot_attention: bool = True
    plot_training_curves: bool = True
    plot_predictions: bool = True
    save_plots: bool = True
    
    # Model checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Results
    results_dir: str = "results"
    save_predictions: bool = True
    save_attention_weights: bool = True
    save_uncertainties: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cuda", "cpu"
    gpu_id: int = 0


# Predefined configurations for different scenarios
CONFIGS = {
    'default': {
        'model': ModelConfig(),
        'training': TrainingConfig(),
        'data': DataConfig(),
        'experiment': ExperimentConfig()
    },
    
    'fast_training': {
        'model': ModelConfig(
            hidden_size=128,
            num_layers=2,
            memory_size=64
        ),
        'training': TrainingConfig(
            epochs=20,
            batch_size=64,
            learning_rate=2e-3
        ),
        'data': DataConfig(
            seq_length=50,
            stride=5,
            sample_size_per_file=1000
        ),
        'experiment': ExperimentConfig()
    },
    
    'high_performance': {
        'model': ModelConfig(
            hidden_size=512,
            num_layers=4,
            num_heads=16,
            memory_size=256,
            hierarchical_levels=4
        ),
        'training': TrainingConfig(
            epochs=100,
            batch_size=16,
            learning_rate=5e-4,
            patience=20
        ),
        'data': DataConfig(
            seq_length=200,
            stride=20
        ),
        'experiment': ExperimentConfig()
    },
    
    'interpretability_focus': {
        'model': ModelConfig(
            hidden_size=256,
            num_layers=3,
            num_heads=8,
            use_temporal_bias=True
        ),
        'training': TrainingConfig(
            epochs=30,
            batch_size=32
        ),
        'data': DataConfig(),
        'experiment': ExperimentConfig(
            plot_attention=True,
            save_attention_weights=True,
            plot_predictions=True
        )
    },
    
    'uncertainty_quantification': {
        'model': ModelConfig(
            use_uncertainty=True,
            use_adversarial=False,
            adaptive_architecture=True
        ),
        'training': TrainingConfig(
            epochs=40,
            learning_rate=1e-3
        ),
        'data': DataConfig(),
        'experiment': ExperimentConfig(
            save_uncertainties=True
        )
    },
    
    'online_learning': {
        'model': ModelConfig(
            adaptive_architecture=True,
            memory_size=256
        ),
        'training': TrainingConfig(
            epochs=30,
            learning_rate=1e-3
        ),
        'data': DataConfig(
            augment_training=True,
            balance_classes=True
        ),
        'experiment': ExperimentConfig()
    }
}


def get_config(config_name: str = 'default') -> Dict:
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Dictionary containing all configuration objects
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. "
                        f"Available configs: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


def create_custom_config(**kwargs) -> Dict:
    """
    Create a custom configuration by overriding default values.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        Custom configuration dictionary
    """
    config = get_config('default').copy()
    
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested attributes like 'model.hidden_size'
            config_type, attr = key.split('.', 1)
            if config_type in config:
                setattr(config[config_type], attr, value)
        else:
            # Handle top-level overrides
            if key in config:
                config[key] = value
    
    return config


def validate_config(config: Dict) -> bool:
    """
    Validate configuration consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # Check data splits sum to 1.0
    total_split = data_config.train_split + data_config.val_split + data_config.test_split
    if not (0.99 <= total_split <= 1.01):
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    # Check sequence length is reasonable
    if data_config.seq_length < 10:
        raise ValueError("Sequence length should be at least 10")
    
    # Check batch size
    if training_config.batch_size < 1:
        raise ValueError("Batch size must be positive")
    
    # Check attention heads divisibility
    if model_config.hidden_size % model_config.num_heads != 0:
        raise ValueError("Hidden size must be divisible by number of attention heads")
    
    return True


# Advanced hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACES = {
    'hidden_size': [128, 256, 512],
    'num_layers': [2, 3, 4],
    'num_heads': [4, 8, 16],
    'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [16, 32, 64],
    'seq_length': [50, 100, 150, 200],
    'memory_size': [64, 128, 256]
}


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_config: Device configuration string
        
    Returns:
        PyTorch device object
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    elif device_config == "cpu":
        return torch.device("cpu")
    else:
        return torch.device(device_config)


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Test default config
    config = get_config('default')
    print(f"Default config loaded: {type(config)}")
    
    # Test validation
    try:
        validate_config(config)
        print("[OK] Default configuration is valid")
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
    
    # Test custom config
    custom_config = create_custom_config(**{
        'model.hidden_size': 512,
        'training.learning_rate': 2e-3,
        'data.seq_length': 150
    })
    print(f"[OK] Custom configuration created")
    
    # Test all predefined configs
    for name in CONFIGS.keys():
        try:
            cfg = get_config(name)
            validate_config(cfg)
            print(f"[OK] Configuration '{name}' is valid")
        except Exception as e:
            print(f"[ERROR] Configuration '{name}' error: {e}")
    
    print("\nConfiguration system test completed!")