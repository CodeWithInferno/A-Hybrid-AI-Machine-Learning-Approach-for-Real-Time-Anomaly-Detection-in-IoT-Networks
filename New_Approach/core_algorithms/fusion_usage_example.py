"""
Revolutionary Game-Theoretic Fusion Layer - Usage Example

This example demonstrates how to use the game-theoretic fusion layer
for robust IoT botnet detection in various scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from game_theoretic_fusion import GameTheoreticFusionLayer, GameType


def simulate_model_outputs(scenario_type: str, n_samples: int = 100) -> tuple:
    """
    Simulate LSTM and ARIMA model outputs for different scenarios
    
    Args:
        scenario_type: Type of scenario ('normal', 'attack', 'evolving')
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (lstm_output, arima_output, context)
    """
    np.random.seed(42)
    
    if scenario_type == 'normal':
        # Normal operation - high confidence, low attack probability
        lstm_output = np.random.beta(8, 2, n_samples)  # High confidence
        arima_output = np.random.beta(7, 3, n_samples)  # Moderate confidence
        context = {
            'attack_probability': 0.1,
            'uncertainty': 0.2,
            'lstm_confidence': 0.9,
            'arima_confidence': 0.85,
            'data_quality': 0.95,
            'attack_history': []
        }
        
    elif scenario_type == 'attack':
        # Under attack - degraded performance, high attack probability
        lstm_output = np.random.beta(3, 7, n_samples)  # Degraded
        arima_output = np.random.beta(4, 6, n_samples)  # Degraded
        context = {
            'attack_probability': 0.8,
            'uncertainty': 0.7,
            'lstm_confidence': 0.6,
            'arima_confidence': 0.65,
            'data_quality': 0.5,
            'attack_history': [0.1, 0.3, 0.7, 0.9, 0.8]
        }
        
    else:  # evolving
        # Evolving attack - progressive degradation
        t = np.random.randint(0, 5)
        attack_intensity = 0.1 + 0.15 * t
        
        lstm_output = np.random.beta(8 - t, 2 + t, n_samples)
        arima_output = np.random.beta(7 - t/2, 3 + t/2, n_samples)
        context = {
            'attack_probability': attack_intensity,
            'uncertainty': 0.3 + 0.1 * t,
            'lstm_confidence': 0.9 - 0.05 * t,
            'arima_confidence': 0.85 - 0.03 * t,
            'data_quality': 0.9 - 0.08 * t,
            'attack_history': [attack_intensity] * (t + 1)
        }
    
    return lstm_output, arima_output, context


def example_basic_usage():
    """Demonstrate basic usage of the fusion layer"""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize fusion layer
    fusion_layer = GameTheoreticFusionLayer()
    
    # Simulate model outputs
    lstm_output, arima_output, context = simulate_model_outputs('normal', 50)
    
    # Perform fusion
    result = fusion_layer.fuse(lstm_output, arima_output, context)
    
    print(f"LSTM output (mean): {np.mean(lstm_output):.3f}")
    print(f"ARIMA output (mean): {np.mean(arima_output):.3f}")
    print(f"Fusion output (mean): {np.mean(result['fusion_output']):.3f}")
    print(f"Final weights: LSTM={result['final_weights'][0]:.3f}, "
          f"ARIMA={result['final_weights'][1]:.3f}")
    print(f"Robustness radius: {result['robustness_radius']:.3f}")


def example_strategy_comparison():
    """Compare different game theory strategies"""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON EXAMPLE")
    print("=" * 60)
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Test different scenarios
    scenarios = ['normal', 'attack', 'evolving']
    
    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} SCENARIO ---")
        
        lstm_output, arima_output, context = simulate_model_outputs(scenario)
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        
        strategies = [
            ('Nash Equilibrium', 'nash_weights'),
            ('Minimax', 'minimax_weights'),
            ('Bayesian Game', 'bayesian_weights'),
            ('Evolutionary', 'evolutionary_weights'),
            ('Stackelberg', 'stackelberg_weights'),
            ('Cooperative', 'cooperative_weights'),
            ('Adversarial Robust', 'robust_weights')
        ]
        
        for name, key in strategies:
            weights = result.get(key, [0, 0])
            if isinstance(weights, np.ndarray) and len(weights) >= 2:
                print(f"  {name:18s}: LSTM={weights[0]:.3f}, ARIMA={weights[1]:.3f}")
        
        print(f"  Final Fusion:       LSTM={result['final_weights'][0]:.3f}, "
              f"ARIMA={result['final_weights'][1]:.3f}")


def example_adaptive_behavior():
    """Demonstrate adaptive behavior over time"""
    print("\n" + "=" * 60)
    print("ADAPTIVE BEHAVIOR EXAMPLE")
    print("=" * 60)
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Simulate evolving attack scenario
    attack_intensities = np.linspace(0.1, 0.9, 10)
    lstm_weights = []
    robustness_values = []
    
    for t, attack_prob in enumerate(attack_intensities):
        # Simulate degrading model performance
        lstm_conf = 0.9 - 0.05 * (attack_prob - 0.1) * 10
        arima_conf = 0.85 - 0.03 * (attack_prob - 0.1) * 10
        
        lstm_output = np.random.beta(max(1, 8 - 5*attack_prob), 2 + 3*attack_prob, 50)
        arima_output = np.random.beta(max(1, 7 - 3*attack_prob), 3 + 2*attack_prob, 50)
        
        context = {
            'attack_probability': attack_prob,
            'uncertainty': 0.2 + 0.5 * attack_prob,
            'lstm_confidence': lstm_conf,
            'arima_confidence': arima_conf,
            'data_quality': 0.95 - 0.4 * attack_prob,
            'attack_history': [attack_prob] * (t + 1)
        }
        
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        
        lstm_weights.append(result['final_weights'][0])
        robustness_values.append(result['robustness_radius'])
        
        print(f"Time {t+1:2d}: Attack={attack_prob:.1f}, "
              f"LSTM_weight={result['final_weights'][0]:.3f}, "
              f"Robustness={result['robustness_radius']:.3f}")
    
    # Plot adaptation over time
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(attack_intensities, lstm_weights, 'b-o', label='LSTM Weight')
    plt.plot(attack_intensities, 1 - np.array(lstm_weights), 'r-s', label='ARIMA Weight')
    plt.xlabel('Attack Intensity')
    plt.ylabel('Fusion Weight')
    plt.title('Adaptive Weight Adjustment')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(attack_intensities, robustness_values, 'g-^')
    plt.xlabel('Attack Intensity')
    plt.ylabel('Robustness Radius')
    plt.title('Robustness vs Attack Intensity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_behavior.png', dpi=150, bbox_inches='tight')
    plt.show()


def example_performance_analysis():
    """Analyze performance under different conditions"""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS EXAMPLE")
    print("=" * 60)
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Test different data sizes
    data_sizes = [10, 50, 100, 500, 1000]
    processing_times = []
    
    import time
    
    for size in data_sizes:
        lstm_output, arima_output, context = simulate_model_outputs('normal', size)
        
        start_time = time.time()
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        print(f"Data size {size:4d}: {processing_time:.4f}s "
              f"({processing_time/size*1000:.2f}ms per sample)")
    
    # Plot performance scaling
    plt.figure(figsize=(8, 5))
    plt.plot(data_sizes, processing_times, 'bo-')
    plt.xlabel('Data Size')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Performance Scaling')
    plt.grid(True)
    plt.savefig('performance_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


def example_robustness_evaluation():
    """Evaluate robustness guarantees"""
    print("\n" + "=" * 60)
    print("ROBUSTNESS EVALUATION EXAMPLE")
    print("=" * 60)
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Test different perturbation levels
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25]
    robustness_radii = []
    
    for epsilon in epsilons:
        # Configure adversarial robustness component
        fusion_layer.adversarial.epsilon = epsilon
        
        lstm_output, arima_output, context = simulate_model_outputs('normal')
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        
        robustness_radii.append(result['robustness_radius'])
        
        print(f"Perturbation budget eps={epsilon:.2f}: "
              f"Certified robustness={result['robustness_radius']:.3f}")
    
    # Plot robustness vs perturbation budget
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, robustness_radii, 'ro-')
    plt.xlabel('Perturbation Budget (eps)')
    plt.ylabel('Certified Robustness Radius')
    plt.title('Robustness Guarantees')
    plt.grid(True)
    plt.savefig('robustness_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def example_visualization():
    """Demonstrate visualization capabilities"""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Generate sequence of fusion results for visualization
    scenarios = ['normal', 'attack', 'normal', 'evolving', 'attack']
    
    for i, scenario in enumerate(scenarios):
        lstm_output, arima_output, context = simulate_model_outputs(scenario)
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        fusion_layer.fusion_history.append(result)
        
        print(f"Step {i+1} ({scenario}): Weights=[{result['final_weights'][0]:.3f}, "
              f"{result['final_weights'][1]:.3f}]")
    
    # Generate visualizations
    print("\nGenerating game dynamics visualization...")
    fusion_layer.visualize_game_dynamics()
    
    print("Generating decision boundaries visualization...")
    fusion_layer.visualize_decision_boundaries()


def main():
    """Run all examples"""
    print("REVOLUTIONARY GAME-THEORETIC FUSION LAYER")
    print("Usage Examples and Demonstrations")
    print("=" * 80)
    
    # Run all examples
    example_basic_usage()
    example_strategy_comparison()
    example_adaptive_behavior()
    example_performance_analysis()
    example_robustness_evaluation()
    example_visualization()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("Generated files:")
    print("- adaptive_behavior.png")
    print("- performance_scaling.png") 
    print("- robustness_analysis.png")
    print("- Game dynamics and decision boundary plots")
    print("=" * 80)


if __name__ == "__main__":
    main()