"""
Simple demonstration of the Game-Theoretic Fusion Layer
"""

import numpy as np
from game_theoretic_fusion import GameTheoreticFusionLayer

def simple_demo():
    """Simple demonstration of fusion layer capabilities"""
    print("Game-Theoretic Fusion Layer - Simple Demo")
    print("=" * 50)
    
    # Initialize fusion layer
    fusion_layer = GameTheoreticFusionLayer()
    
    # Scenario 1: Normal conditions
    print("\n1. Normal Conditions:")
    lstm_normal = np.array([0.8, 0.9, 0.7, 0.85, 0.82])
    arima_normal = np.array([0.75, 0.8, 0.85, 0.7, 0.78])
    context_normal = {
        'attack_probability': 0.1,
        'lstm_confidence': 0.9,
        'arima_confidence': 0.85
    }
    
    result1 = fusion_layer.fuse(lstm_normal, arima_normal, context_normal)
    print(f"  LSTM mean: {np.mean(lstm_normal):.3f}")
    print(f"  ARIMA mean: {np.mean(arima_normal):.3f}")
    print(f"  Fusion mean: {np.mean(result1['fusion_output']):.3f}")
    print(f"  Weights: LSTM={result1['final_weights'][0]:.3f}, ARIMA={result1['final_weights'][1]:.3f}")
    print(f"  Robustness: {result1['robustness_radius']:.3f}")
    
    # Scenario 2: Attack conditions
    print("\n2. Under Attack:")
    lstm_attack = np.array([0.4, 0.3, 0.5, 0.2, 0.35])
    arima_attack = np.array([0.45, 0.5, 0.4, 0.6, 0.48])
    context_attack = {
        'attack_probability': 0.8,
        'lstm_confidence': 0.6,
        'arima_confidence': 0.65,
        'attack_history': [0.2, 0.5, 0.8]
    }
    
    result2 = fusion_layer.fuse(lstm_attack, arima_attack, context_attack)
    print(f"  LSTM mean: {np.mean(lstm_attack):.3f}")
    print(f"  ARIMA mean: {np.mean(arima_attack):.3f}")
    print(f"  Fusion mean: {np.mean(result2['fusion_output']):.3f}")
    print(f"  Weights: LSTM={result2['final_weights'][0]:.3f}, ARIMA={result2['final_weights'][1]:.3f}")
    print(f"  Robustness: {result2['robustness_radius']:.3f}")
    
    # Show strategy comparison
    print("\n3. Strategy Comparison (Attack Scenario):")
    strategies = [
        ('Nash Equilibrium', result2.get('nash_weights', [0.5, 0.5])),
        ('Minimax', result2.get('minimax_weights', [0.5, 0.5])),
        ('Evolutionary', result2.get('evolutionary_weights', [0.5, 0.5])),
        ('Cooperative', result2.get('cooperative_weights', [0.5, 0.5])),
        ('Robust', result2.get('robust_weights', [0.5, 0.5]))
    ]
    
    for name, weights in strategies:
        if len(weights) >= 2:
            print(f"  {name:15s}: LSTM={weights[0]:.3f}, ARIMA={weights[1]:.3f}")
    
    print(f"\n  Final Fusion:     LSTM={result2['final_weights'][0]:.3f}, ARIMA={result2['final_weights'][1]:.3f}")
    
    # Performance analysis
    print("\n4. Performance Analysis:")
    import time
    
    sizes = [10, 50, 100, 500]
    for size in sizes:
        lstm_data = np.random.random(size)
        arima_data = np.random.random(size)
        context = {'attack_probability': 0.3}
        
        start_time = time.time()
        result = fusion_layer.fuse(lstm_data, arima_data, context)
        end_time = time.time()
        
        print(f"  Size {size:3d}: {(end_time - start_time)*1000:.1f}ms")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("Key features demonstrated:")
    print("- Multiple game theory strategies")
    print("- Adaptive weight adjustment")
    print("- Robustness guarantees")
    print("- Efficient performance")

if __name__ == "__main__":
    simple_demo()