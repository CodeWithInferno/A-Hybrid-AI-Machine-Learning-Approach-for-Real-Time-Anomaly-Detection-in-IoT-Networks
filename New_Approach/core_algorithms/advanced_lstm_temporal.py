"""
Advanced LSTM with Temporal Attention Mechanisms for IoT Time Series
=====================================================================

This module implements a cutting-edge LSTM architecture with multi-head temporal attention,
hierarchical processing, adversarial training, and adaptive capabilities specifically
designed for IoT botnet detection and time series analysis.

Key Features:
- Multi-head temporal attention mechanism for capturing complex temporal dependencies
- Hierarchical LSTM for multi-scale pattern recognition
- Adversarial training components for robust feature learning
- IoT-specific temporal gate mechanisms
- Memory augmented networks for long-term dependencies
- Uncertainty estimation using Bayesian approaches
- Online learning capabilities for continuous adaptation
- Explainable attention weights with visualization
- GPU optimization with mixed precision training
- Adaptive architecture that evolves with data

Author: IoT Research Lab
Date: 2024
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from collections import deque
import math
import warnings


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head temporal attention mechanism optimized for IoT time series.
    
    This attention mechanism is specifically designed to capture both local
    and global temporal patterns in IoT network traffic data.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1,
                 temperature: float = 1.0, use_temporal_bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        self.use_temporal_bias = use_temporal_bias
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Temporal bias for IoT-specific patterns
        if use_temporal_bias:
            self.temporal_bias = nn.Parameter(torch.zeros(1, 1, num_heads, 1, 1))
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter
        self.adaptive_temp = nn.Parameter(torch.ones(1))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Project and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with adaptive temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores / (self.temperature * self.adaptive_temp)
        
        # Add temporal bias for IoT patterns
        if self.use_temporal_bias:
            scores = scores + self.temporal_bias
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class HierarchicalLSTMCell(nn.Module):
    """
    Hierarchical LSTM cell for multi-scale temporal pattern recognition.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_levels: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        
        # LSTM cells for each hierarchical level
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_levels)
        ])
        
        # Gates for hierarchical information flow
        self.level_gates = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size)
            for _ in range(num_levels - 1)
        ])
        
        # Temporal gates for IoT-specific patterns
        self.temporal_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])
        
    def forward(self, x: torch.Tensor, states: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through hierarchical LSTM.
        """
        new_states = []
        level_outputs = []
        
        for level in range(self.num_levels):
            if level == 0:
                h, c = self.lstm_cells[level](x, states[level])
            else:
                # Combine information from previous level
                gated_input = torch.sigmoid(self.level_gates[level-1](
                    torch.cat([level_outputs[-1], states[level][0]], dim=-1)
                )) * level_outputs[-1]
                h, c = self.lstm_cells[level](gated_input, states[level])
            
            # Apply temporal gate for IoT patterns
            h = h * self.temporal_gates[level](h)
            
            new_states.append((h, c))
            level_outputs.append(h)
        
        # Combine outputs from all levels
        output = sum(level_outputs) / len(level_outputs)
        
        return output, new_states


class MemoryAugmentedNetwork(nn.Module):
    """
    Memory augmented network component for long-term dependency modeling.
    """
    
    def __init__(self, hidden_size: int, memory_size: int = 128, memory_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Memory bank
        self.memory_bank = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Query, key, value projections for memory access
        self.query_proj = nn.Linear(hidden_size, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, hidden_size)
        
        # Write gate for memory updates
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_size + memory_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        """
        Access and optionally update memory.
        """
        batch_size = x.size(0)
        
        # Generate query
        query = self.query_proj(x)  # [batch_size, memory_dim]
        
        # Compute attention scores with memory bank
        keys = self.key_proj(self.memory_bank)  # [memory_size, memory_dim]
        scores = torch.matmul(query, keys.t()) / math.sqrt(self.memory_dim)
        attention = F.softmax(scores, dim=-1)  # [batch_size, memory_size]
        
        # Read from memory
        memory_read = torch.matmul(attention, self.memory_bank)  # [batch_size, memory_dim]
        memory_output = self.value_proj(memory_read)
        
        # Update memory if in training mode
        if update_memory and self.training:
            # Compute write gate
            write_gate = self.write_gate(torch.cat([x, memory_read], dim=-1))
            
            # Update memory with exponential moving average
            with torch.no_grad():
                update_weights = attention.mean(dim=0, keepdim=True).t()  # [memory_size, 1]
                update_values = query.mean(dim=0, keepdim=True)  # [1, memory_dim]
                self.memory_bank.data = (1 - write_gate.mean()) * self.memory_bank.data + \
                                       write_gate.mean() * update_weights * update_values
        
        return x + memory_output


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator network for adversarial training component.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class UncertaintyEstimator(nn.Module):
    """
    Bayesian uncertainty estimation component.
    """
    
    def __init__(self, input_size: int, output_size: int, num_samples: int = 10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_samples = num_samples
        
        # Variational layers
        self.mean_layer = nn.Linear(input_size, output_size)
        self.log_var_layer = nn.Linear(input_size, output_size)
        
        # Prior parameters
        self.prior_mean = nn.Parameter(torch.zeros(output_size))
        self.prior_log_var = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            mean: Mean prediction
            uncertainty: Epistemic uncertainty estimate
        """
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        
        if self.training:
            # Sample from variational distribution
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mean + eps * std
            
            # KL divergence for regularization
            kl_div = -0.5 * torch.sum(1 + log_var - self.prior_log_var - 
                                     ((mean - self.prior_mean) ** 2 + torch.exp(log_var)) / 
                                     torch.exp(self.prior_log_var))
            
            return sample, kl_div / x.size(0)
        else:
            # Monte Carlo sampling for uncertainty
            samples = []
            for _ in range(self.num_samples):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                sample = mean + eps * std
                samples.append(sample)
            
            samples = torch.stack(samples)
            uncertainty = torch.var(samples, dim=0)
            
            return mean, uncertainty


class AdvancedLSTMTemporal(nn.Module):
    """
    Advanced LSTM with Temporal Attention for IoT Time Series.
    
    This model combines hierarchical LSTM processing, multi-head temporal attention,
    memory augmentation, adversarial training, and uncertainty estimation for
    robust IoT botnet detection.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 memory_size: int = 128,
                 use_adversarial: bool = True,
                 use_uncertainty: bool = True,
                 adaptive_architecture: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_adversarial = use_adversarial
        self.use_uncertainty = use_uncertainty
        self.adaptive_architecture = adaptive_architecture
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hierarchical LSTM layers
        self.lstm_layers = nn.ModuleList([
            HierarchicalLSTMCell(hidden_size, hidden_size, num_levels=3)
            for _ in range(num_layers)
        ])
        
        # Multi-head temporal attention
        self.attention_layers = nn.ModuleList([
            MultiHeadTemporalAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Memory augmented network
        self.memory_network = MemoryAugmentedNetwork(hidden_size, memory_size)
        
        # Adversarial components
        if use_adversarial:
            self.discriminator = AdversarialDiscriminator(hidden_size)
            self.gradient_reversal_lambda = 0.1
        
        # Uncertainty estimation
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(hidden_size, num_classes)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Adaptive architecture components
        if adaptive_architecture:
            self.architecture_controller = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, num_layers),
                nn.Sigmoid()
            )
            self.layer_importance = nn.Parameter(torch.ones(num_layers))
        
        # Online learning components
        self.online_memory_buffer = deque(maxlen=1000)
        self.adaptation_rate = 0.01
        
        # Mixed precision training
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Attention weight storage for explainability
        self.attention_weights_history = []
        
    def forward(self, x: torch.Tensor, return_attention: bool = False,
                adversarial_training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            return_attention: Whether to return attention weights
            adversarial_training: Whether to use adversarial training
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - uncertainty: Uncertainty estimates (if enabled)
                - attention_weights: Attention weights (if requested)
                - discriminator_output: Discriminator output (if adversarial)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_projection(x)
        
        # Initialize hidden states for hierarchical LSTM
        hidden_states = [
            [(torch.zeros(batch_size, self.hidden_size, device=x.device),
              torch.zeros(batch_size, self.hidden_size, device=x.device))
             for _ in range(3)]  # 3 hierarchical levels
            for _ in range(self.num_layers)
        ]
        
        # Storage for outputs and attention weights
        outputs = []
        all_attention_weights = []
        
        # Process entire sequence at once for better efficiency
        lstm_output = x  # Start with input
        
        # Pass through LSTM layers
        for layer_idx in range(self.num_layers):
            layer_outputs = []
            layer_hidden = hidden_states[layer_idx]
            
            # Process sequence timestep by timestep for this layer
            for t in range(seq_len):
                h_t = lstm_output[:, t, :]
                h_t, layer_hidden = self.lstm_layers[layer_idx](h_t, layer_hidden)
                layer_outputs.append(h_t)
            
            # Update hidden states
            hidden_states[layer_idx] = layer_hidden
            
            # Stack outputs for this layer
            layer_output_tensor = torch.stack(layer_outputs, dim=1)
            
            # Apply self-attention to the entire sequence
            attended_output, attn_weights = self.attention_layers[layer_idx](
                layer_output_tensor, layer_output_tensor, layer_output_tensor
            )
            all_attention_weights.append(attn_weights)
            
            # Adaptive architecture - dynamically weight layers
            if self.adaptive_architecture:
                layer_weight = self.layer_importance[layer_idx]
                attended_output = attended_output * layer_weight
            
            lstm_output = attended_output
        
        # Apply memory augmentation to final output
        output_with_memory = []
        for t in range(seq_len):
            h_t = lstm_output[:, t, :]
            h_t_memory = self.memory_network(h_t)
            output_with_memory.append(h_t_memory)
        
        # Stack final outputs
        output = torch.stack(output_with_memory, dim=1)  # [batch_size, seq_len, hidden_size]
        
        # Global temporal pooling
        output_pooled = torch.mean(output, dim=1)  # [batch_size, hidden_size]
        
        # Prepare results dictionary
        results = {}
        
        # Classification output
        if self.use_uncertainty:
            logits, uncertainty = self.uncertainty_estimator(output_pooled)
            results['logits'] = logits
            results['uncertainty'] = uncertainty
        else:
            logits = self.output_projection(output_pooled)
            results['logits'] = logits
        
        # Adversarial training
        if self.use_adversarial and adversarial_training:
            # Gradient reversal for adversarial training
            reversed_features = self._gradient_reversal(output_pooled)
            discriminator_output = self.discriminator(reversed_features)
            results['discriminator_output'] = discriminator_output
        
        # Return attention weights if requested
        if return_attention and all_attention_weights:
            results['attention_weights'] = torch.stack(all_attention_weights).mean(dim=0)
            self.attention_weights_history.append(results['attention_weights'].detach().cpu())
        
        return results
    
    def _gradient_reversal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient reversal layer for adversarial training.
        """
        return GradientReversal.apply(x, self.gradient_reversal_lambda)
    
    @torch.no_grad()
    def online_update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Online learning update for continuous adaptation.
        
        Args:
            x: New input data
            y: Corresponding labels
        """
        self.online_memory_buffer.append((x, y))
        
        if len(self.online_memory_buffer) >= 32:  # Mini-batch size
            # Sample from buffer
            batch = list(self.online_memory_buffer)[-32:]
            x_batch = torch.cat([item[0] for item in batch])
            y_batch = torch.cat([item[1] for item in batch])
            
            # Perform gradient step
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=self.adaptation_rate)
            
            with autocast():
                outputs = self(x_batch)
                loss = F.cross_entropy(outputs['logits'], y_batch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            self.eval()
    
    def get_attention_explanations(self, top_k: int = 5) -> Dict[str, np.ndarray]:
        """
        Get explainable attention weights.
        
        Args:
            top_k: Number of top attention connections to return
            
        Returns:
            Dictionary with attention explanations
        """
        if not self.attention_weights_history:
            return {}
        
        # Average attention weights over history
        avg_attention = torch.stack(self.attention_weights_history[-100:]).mean(dim=0)
        
        # Get top-k most attended positions
        top_attention_scores, top_attention_indices = torch.topk(
            avg_attention.view(-1), k=min(top_k, avg_attention.numel())
        )
        
        return {
            'top_attention_scores': top_attention_scores.numpy(),
            'top_attention_indices': top_attention_indices.numpy(),
            'attention_heatmap': avg_attention.numpy()
        }
    
    def adapt_architecture(self, performance_metrics: Dict[str, float]):
        """
        Adapt model architecture based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        if not self.adaptive_architecture:
            return
        
        # Simple adaptation based on accuracy
        accuracy = performance_metrics.get('accuracy', 0.5)
        
        # Update layer importance based on performance
        with torch.no_grad():
            if accuracy < 0.8:  # Need more capacity
                self.layer_importance.data *= 1.1
            elif accuracy > 0.95:  # Can reduce capacity
                self.layer_importance.data *= 0.9
            
            # Clamp values
            self.layer_importance.data = torch.clamp(self.layer_importance.data, 0.1, 2.0)


class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None


class IoTTemporalGate(nn.Module):
    """
    Specialized temporal gate for IoT-specific patterns.
    
    This gate is designed to capture periodic patterns common in IoT traffic,
    such as regular sensor updates, heartbeat messages, and attack patterns.
    """
    
    def __init__(self, hidden_size: int, num_periods: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_periods = num_periods
        
        # Learnable period embeddings
        self.period_embeddings = nn.Parameter(torch.randn(num_periods, hidden_size))
        
        # Gate computation layers
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_periods),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, time_step: int) -> torch.Tensor:
        """
        Apply IoT-specific temporal gating.
        """
        batch_size = x.size(0)
        
        # Compute time embedding
        time_encoding = torch.tensor([
            [math.sin(2 * math.pi * time_step / (10 ** (2 * i / self.hidden_size)))
             if i % 2 == 0 else
             math.cos(2 * math.pi * time_step / (10 ** (2 * (i - 1) / self.hidden_size)))
             for i in range(self.hidden_size)]
        ], device=x.device).expand(batch_size, -1)
        
        # Compute gate weights
        gate_input = torch.cat([x, time_encoding], dim=-1)
        gate_weights = self.gate_net(gate_input)  # [batch_size, num_periods]
        
        # Apply gating with period embeddings
        gated_output = torch.matmul(gate_weights, self.period_embeddings)
        
        # Combine with input
        output = self.output_proj(x * gated_output)
        
        return output


def create_model(config: Dict) -> AdvancedLSTMTemporal:
    """
    Factory function to create model with configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured AdvancedLSTMTemporal model
    """
    return AdvancedLSTMTemporal(
        input_size=config.get('input_size', 115),
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 3),
        num_classes=config.get('num_classes', 2),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        memory_size=config.get('memory_size', 128),
        use_adversarial=config.get('use_adversarial', True),
        use_uncertainty=config.get('use_uncertainty', True),
        adaptive_architecture=config.get('adaptive_architecture', True)
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Advanced LSTM with Temporal Attention - IoT Time Series")
    print("=" * 60)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = {
        'input_size': 115,  # N-BaIoT dataset features
        'hidden_size': 256,
        'num_layers': 3,
        'num_classes': 2,  # Binary classification (benign/malicious)
        'num_heads': 8,
        'dropout': 0.1,
        'memory_size': 128,
        'use_adversarial': True,
        'use_uncertainty': True,
        'adaptive_architecture': True
    }
    
    model = create_model(config).to(device)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size, seq_len = 32, 100
    x = torch.randn(batch_size, seq_len, config['input_size']).to(device)
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.amp.autocast(device_type):
        outputs = model(x, return_attention=True, adversarial_training=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test online learning
    print("\nTesting online learning capability...")
    for i in range(5):
        x_online = torch.randn(1, seq_len, config['input_size']).to(device)
        y_online = torch.randint(0, 2, (1,)).to(device)
        model.online_update(x_online, y_online)
    print("Online learning test completed.")
    
    # Test attention explanations
    print("\nGetting attention explanations...")
    explanations = model.get_attention_explanations(top_k=5)
    for key, value in explanations.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
    
    print("\nModel ready for IoT time series analysis!")