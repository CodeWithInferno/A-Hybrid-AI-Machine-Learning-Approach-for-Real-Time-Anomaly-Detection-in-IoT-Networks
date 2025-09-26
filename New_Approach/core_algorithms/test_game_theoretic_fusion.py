"""
Test Suite for Game-Theoretic Fusion Layer

Comprehensive tests to validate all game theory paradigms and mathematical foundations
"""

import numpy as np
import torch
from game_theoretic_fusion import (
    GameTheoreticFusionLayer, NashEquilibriumSolver, MinimaxStrategy,
    BayesianGameModel, EvolutionaryGameDynamics, StackelbergGame,
    CooperativeGameTheory, MechanismDesign, RegretMinimization,
    MultiAgentRL, AdversarialRobustness, GameState
)


class TestNashEquilibriumSolver:
    """Test Nash equilibrium computation"""
    
    def test_payoff_matrix_computation(self):
        solver = NashEquilibriumSolver()
        payoff = solver.compute_payoff_matrix(0.8, 0.7, 0.3)
        
        assert payoff.shape == (10, 10)
        assert np.all(np.isfinite(payoff))
        
    def test_nash_equilibrium_computation(self):
        solver = NashEquilibriumSolver()
        payoff = np.array([[3, 0], [0, 1]])
        
        defender, attacker = solver.find_nash_equilibrium(payoff)
        
        assert len(defender) == 2
        assert len(attacker) == 2
        assert np.allclose(np.sum(defender), 1, atol=1e-6)
        assert np.allclose(np.sum(attacker), 1, atol=1e-6)


class TestMinimaxStrategy:
    """Test minimax strategy implementation"""
    
    def test_worst_case_fusion(self):
        minimax = MinimaxStrategy()
        lstm_output = np.array([0.8, 0.7, 0.9])
        arima_output = np.array([0.6, 0.8, 0.7])
        
        scenarios = [
            {'lstm_degradation': 0.2, 'arima_degradation': 0.1},
            {'lstm_degradation': 0.1, 'arima_degradation': 0.3}
        ]
        
        weights = minimax.compute_worst_case_fusion(lstm_output, arima_output, scenarios)
        
        assert len(weights) == 2
        assert np.allclose(np.sum(weights), 1, atol=1e-6)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)


class TestBayesianGameModel:
    """Test Bayesian game model"""
    
    def test_belief_update(self):
        bayesian = BayesianGameModel(n_types=3)
        observations = np.array([0.1, 0.2, 0.3, 0.4])
        
        updated_beliefs = bayesian.update_beliefs(observations)
        
        assert len(updated_beliefs) == 3
        assert np.allclose(np.sum(updated_beliefs), 1, atol=1e-6)
        assert np.all(updated_beliefs >= 0)
        
    def test_bayesian_fusion(self):
        bayesian = BayesianGameModel()
        lstm_output = np.array([0.8, 0.7])
        arima_output = np.array([0.6, 0.8])
        
        weights = bayesian.compute_bayesian_fusion(lstm_output, arima_output)
        
        assert len(weights) == 2
        assert np.allclose(np.sum(weights), 1, atol=1e-6)


class TestEvolutionaryGameDynamics:
    """Test evolutionary game dynamics"""
    
    def test_population_initialization(self):
        evo = EvolutionaryGameDynamics(population_size=50)
        
        assert evo.population.shape == (50, 2)
        assert np.allclose(np.sum(evo.population, axis=1), 1, atol=1e-6)
        
    def test_fitness_function(self):
        evo = EvolutionaryGameDynamics()
        strategy = np.array([0.6, 0.4])
        environment = {'lstm_accuracy': 0.8, 'arima_accuracy': 0.7, 'attack_intensity': 0.3}
        
        fitness = evo.fitness_function(strategy, environment)
        
        assert isinstance(fitness, (int, float))
        assert fitness >= 0
        
    def test_evolution(self):
        evo = EvolutionaryGameDynamics(population_size=20)
        environment = {'lstm_accuracy': 0.8, 'arima_accuracy': 0.7, 'attack_intensity': 0.3}
        
        best_strategy = evo.evolve(environment, generations=5)
        
        assert len(best_strategy) == 2
        assert np.allclose(np.sum(best_strategy), 1, atol=1e-6)


class TestStackelbergGame:
    """Test Stackelberg game implementation"""
    
    def test_stackelberg_equilibrium(self):
        stackelberg = StackelbergGame()
        lstm_params = {'accuracy': 0.8, 'detection_rate': 0.9}
        arima_params = {'accuracy': 0.7, 'detection_rate': 0.85}
        
        leader, follower = stackelberg.compute_stackelberg_equilibrium(lstm_params, arima_params)
        
        assert len(leader) == 2
        assert len(follower) == 2
        assert np.allclose(np.sum(leader), 1, atol=1e-6)


class TestCooperativeGameTheory:
    """Test cooperative game theory"""
    
    def test_shapley_values(self):
        coop = CooperativeGameTheory()
        contributions = {'lstm': 0.8, 'arima': 0.7}
        
        shapley = coop.compute_shapley_values(contributions)
        
        assert 'lstm' in shapley
        assert 'arima' in shapley
        assert all(isinstance(v, (int, float)) for v in shapley.values())
        
    def test_core_allocation(self):
        coop = CooperativeGameTheory()
        
        weights = coop.compute_core_allocation(0.8, 0.7)
        
        assert len(weights) == 2
        assert np.allclose(np.sum(weights), 1, atol=1e-6)
        assert np.all(weights >= 0)


class TestMechanismDesign:
    """Test mechanism design"""
    
    def test_vickrey_auction(self):
        mechanism = MechanismDesign(budget=1.0)
        bids = {'lstm': 0.8, 'arima': 0.7}
        
        payments = mechanism.design_vickrey_auction(bids)
        
        assert 'lstm' in payments
        assert 'arima' in payments
        assert payments['lstm'] >= 0
        assert payments['arima'] >= 0
        
    def test_incentive_compatible_weights(self):
        mechanism = MechanismDesign()
        
        weights = mechanism.compute_incentive_compatible_weights(0.8, 0.7)
        
        assert len(weights) == 2
        assert np.allclose(np.sum(weights), 1, atol=1e-6)


class TestRegretMinimization:
    """Test regret minimization algorithms"""
    
    def test_regret_update(self):
        regret_min = RegretMinimization()
        actions = np.array([0.6, 0.4])
        rewards = np.array([0.8, 0.7])
        
        regret_min.update_regrets(actions, rewards)
        
        assert regret_min.t == 1
        assert len(regret_min.cumulative_regret) == 2
        
    def test_strategy_computation(self):
        regret_min = RegretMinimization()
        
        strategy = regret_min.get_strategy()
        
        assert len(strategy) == 2
        assert np.allclose(np.sum(strategy), 1, atol=1e-6)


class TestMultiAgentRL:
    """Test multi-agent reinforcement learning"""
    
    def test_forward_pass(self):
        marl = MultiAgentRL(state_dim=10)
        state = torch.randn(1, 10)
        
        lstm_weight, arima_weight = marl(state)
        
        assert lstm_weight.shape == (1, 1)
        assert arima_weight.shape == (1, 1)
        assert torch.allclose(lstm_weight + arima_weight, torch.ones(1, 1))
        
    def test_value_function(self):
        marl = MultiAgentRL(state_dim=10)
        state = torch.randn(1, 10)
        actions = torch.randn(1, 2)
        
        value = marl.get_value(state, actions)
        
        assert value.shape == (1, 1)
        assert torch.isfinite(value).all()


class TestAdversarialRobustness:
    """Test adversarial robustness guarantees"""
    
    def test_robust_weights_computation(self):
        robust = AdversarialRobustness(epsilon=0.1)
        lstm_output = np.array([0.8, 0.7, 0.9])
        arima_output = np.array([0.6, 0.8, 0.7])
        
        weights = robust.compute_robust_weights(lstm_output, arima_output)
        
        assert len(weights) == 2
        assert np.allclose(np.sum(weights), 1, atol=1e-6)
        
    def test_robustness_certification(self):
        robust = AdversarialRobustness(epsilon=0.1)
        weights = np.array([0.6, 0.4])
        lstm_output = np.array([0.8, 0.7])
        arima_output = np.array([0.6, 0.8])
        
        radius = robust.certify_robustness(weights, lstm_output, arima_output)
        
        assert isinstance(radius, (int, float))
        assert radius >= 0


class TestGameTheoreticFusionLayer:
    """Test the complete fusion layer"""
    
    def test_fusion_layer_initialization(self):
        fusion_layer = GameTheoreticFusionLayer()
        
        assert fusion_layer.nash_solver is not None
        assert fusion_layer.minimax is not None
        assert fusion_layer.bayesian_game is not None
        assert fusion_layer.evolutionary is not None
        assert fusion_layer.stackelberg is not None
        assert fusion_layer.cooperative is not None
        assert fusion_layer.mechanism_design is not None
        assert fusion_layer.regret_min is not None
        assert fusion_layer.marl is not None
        assert fusion_layer.adversarial is not None
        
    def test_fusion_process(self):
        fusion_layer = GameTheoreticFusionLayer()
        lstm_output = np.array([0.8, 0.7, 0.9, 0.6, 0.8])
        arima_output = np.array([0.6, 0.8, 0.7, 0.9, 0.5])
        
        context = {
            'attack_probability': 0.3,
            'uncertainty': 0.4,
            'lstm_confidence': 0.85,
            'arima_confidence': 0.80,
            'data_quality': 0.9,
            'attack_history': [0.1, 0.2, 0.3]
        }
        
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        
        # Check result structure
        assert 'fusion_output' in result
        assert 'final_weights' in result
        assert 'nash_weights' in result
        assert 'minimax_weights' in result
        assert 'bayesian_weights' in result
        assert 'evolutionary_weights' in result
        assert 'stackelberg_weights' in result
        assert 'cooperative_weights' in result
        assert 'incentive_weights' in result
        assert 'regret_weights' in result
        assert 'marl_weights' in result
        assert 'robust_weights' in result
        assert 'robustness_radius' in result
        assert 'shapley_values' in result
        assert 'game_state' in result
        
        # Check weight properties
        assert len(result['final_weights']) == 2
        assert np.allclose(np.sum(result['final_weights']), 1, atol=1e-6)
        assert np.all(result['final_weights'] >= 0)
        assert np.all(result['final_weights'] <= 1)
        
        # Check fusion output
        assert len(result['fusion_output']) == len(lstm_output)
        assert np.all(np.isfinite(result['fusion_output']))
        
    def test_meta_weights_computation(self):
        fusion_layer = GameTheoreticFusionLayer()
        
        # High attack scenario
        context_attack = {
            'attack_probability': 0.8,
            'uncertainty': 0.7,
            'data_quality': 0.6
        }
        
        meta_weights = fusion_layer._compute_meta_weights(context_attack)
        
        assert len(meta_weights) == 10
        assert np.allclose(np.sum(meta_weights), 1, atol=1e-6)
        assert np.all(meta_weights >= 0)
        
        # Normal scenario
        context_normal = {
            'attack_probability': 0.1,
            'uncertainty': 0.2,
            'data_quality': 0.9
        }
        
        meta_weights_normal = fusion_layer._compute_meta_weights(context_normal)
        
        # Should be different from attack scenario
        assert not np.allclose(meta_weights, meta_weights_normal)
        
    def test_game_state_tracking(self):
        fusion_layer = GameTheoreticFusionLayer()
        lstm_output = np.array([0.8, 0.7])
        arima_output = np.array([0.6, 0.8])
        
        # Perform multiple fusions
        for i in range(3):
            context = {
                'attack_probability': 0.1 + 0.2 * i,
                'uncertainty': 0.2 + 0.1 * i
            }
            fusion_layer.fuse(lstm_output, arima_output, context)
            
        assert len(fusion_layer.game_states) == 3
        
        for state in fusion_layer.game_states:
            assert isinstance(state, GameState)
            assert state.lstm_output is not None
            assert state.arima_output is not None
            assert state.fusion_weights is not None


def run_performance_tests():
    """Run performance tests for the fusion layer"""
    import time
    
    print("Running performance tests...")
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Test with different data sizes
    sizes = [10, 50, 100, 500]
    
    for size in sizes:
        lstm_output = np.random.beta(8, 2, size)
        arima_output = np.random.beta(7, 3, size)
        
        context = {
            'attack_probability': 0.3,
            'uncertainty': 0.4,
            'lstm_confidence': 0.85,
            'arima_confidence': 0.80
        }
        
        start_time = time.time()
        result = fusion_layer.fuse(lstm_output, arima_output, context)
        end_time = time.time()
        
        print(f"Size {size}: {end_time - start_time:.4f} seconds")
        
        # Verify result quality
        assert np.all(np.isfinite(result['fusion_output']))
        assert 0 <= result['robustness_radius'] <= 1


def run_stress_tests():
    """Run stress tests with extreme conditions"""
    print("Running stress tests...")
    
    fusion_layer = GameTheoreticFusionLayer()
    
    # Test with extreme values
    test_cases = [
        {
            'name': 'All zeros',
            'lstm': np.zeros(10),
            'arima': np.zeros(10),
            'context': {'attack_probability': 0.5}
        },
        {
            'name': 'All ones',
            'lstm': np.ones(10),
            'arima': np.ones(10),
            'context': {'attack_probability': 0.5}
        },
        {
            'name': 'High attack probability',
            'lstm': np.random.random(10),
            'arima': np.random.random(10),
            'context': {'attack_probability': 0.99}
        },
        {
            'name': 'High uncertainty',
            'lstm': np.random.random(10),
            'arima': np.random.random(10),
            'context': {'uncertainty': 0.99}
        },
        {
            'name': 'Very different model outputs',
            'lstm': np.ones(10) * 0.9,
            'arima': np.ones(10) * 0.1,
            'context': {'attack_probability': 0.5}
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        
        try:
            result = fusion_layer.fuse(
                test_case['lstm'], 
                test_case['arima'], 
                test_case['context']
            )
            
            # Verify result is valid
            assert np.all(np.isfinite(result['fusion_output']))
            assert len(result['final_weights']) == 2
            assert np.allclose(np.sum(result['final_weights']), 1, atol=1e-6)
            
            print(f"  PASS: weights={result['final_weights']}")
            
        except Exception as e:
            print(f"  FAIL: {e}")
            raise


if __name__ == "__main__":
    # Run all tests
    print("Running Game-Theoretic Fusion Layer Tests")
    print("=" * 50)
    
    # Unit tests
    test_classes = [
        TestNashEquilibriumSolver(),
        TestMinimaxStrategy(),
        TestBayesianGameModel(),
        TestEvolutionaryGameDynamics(),
        TestStackelbergGame(),
        TestCooperativeGameTheory(),
        TestMechanismDesign(),
        TestRegretMinimization(),
        TestMultiAgentRL(),
        TestAdversarialRobustness(),
        TestGameTheoreticFusionLayer()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\nTesting {class_name}:")
        
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  PASS {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  FAIL {method_name}: {e}")
    
    print(f"\nUnit Tests: {passed_tests}/{total_tests} passed")
    
    # Performance tests
    print("\n" + "=" * 50)
    run_performance_tests()
    
    # Stress tests
    print("\n" + "=" * 50)
    run_stress_tests()
    
    print("\n" + "=" * 50)
    print("All tests completed!")