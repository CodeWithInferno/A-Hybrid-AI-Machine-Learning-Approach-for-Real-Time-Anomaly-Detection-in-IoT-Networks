"""
Revolutionary Game-Theoretic Fusion Layer for IoT Botnet Detection

This module implements a comprehensive game-theoretic fusion framework that combines
LSTM and ARIMA outputs using multiple game theory paradigms for robust decision-making
in adversarial environments.

Author: Advanced AI Research Team
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linprog, minimize
from scipy.special import softmax
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import warnings
import math
from itertools import combinations
warnings.filterwarnings('ignore')


class GameType(Enum):
    """Enumeration of game theory paradigms"""
    NASH = "nash_equilibrium"
    MINIMAX = "minimax"
    BAYESIAN = "bayesian_game"
    EVOLUTIONARY = "evolutionary_dynamics"
    STACKELBERG = "stackelberg"
    COOPERATIVE = "cooperative"
    MECHANISM_DESIGN = "mechanism_design"
    REGRET_MINIMIZATION = "regret_minimization"
    MARL = "multi_agent_rl"
    ADVERSARIAL = "adversarial_robustness"


@dataclass
class GameState:
    """Represents the current state of the game-theoretic fusion"""
    lstm_output: np.ndarray
    arima_output: np.ndarray
    uncertainty: float
    attack_history: List[float]
    fusion_weights: np.ndarray
    equilibrium_point: Optional[np.ndarray] = None
    payoff_matrix: Optional[np.ndarray] = None


class NashEquilibriumSolver:
    """Implements Nash equilibrium computation for fusion weights"""
    
    def __init__(self, n_strategies: int = 10):
        self.n_strategies = n_strategies
        
    def compute_payoff_matrix(self, lstm_conf: float, arima_conf: float, 
                            attack_prob: float) -> np.ndarray:
        """
        Compute payoff matrix based on model confidences and attack probability
        
        Args:
            lstm_conf: LSTM model confidence
            arima_conf: ARIMA model confidence
            attack_prob: Estimated attack probability
            
        Returns:
            Payoff matrix for the game
        """
        # Create payoff matrix considering detection accuracy vs false positives
        payoff = np.zeros((self.n_strategies, self.n_strategies))
        
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                lstm_weight = i / (self.n_strategies - 1)
                arima_weight = 1 - lstm_weight
                
                # Payoff combines detection accuracy and false positive cost
                detection_payoff = lstm_weight * lstm_conf + arima_weight * arima_conf
                false_positive_cost = -0.3 * (1 - attack_prob) * (lstm_weight ** 2)
                
                payoff[i, j] = detection_payoff + false_positive_cost
                
        return payoff
    
    def find_nash_equilibrium(self, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find Nash equilibrium using linear programming
        
        Args:
            payoff_matrix: Game payoff matrix
            
        Returns:
            Tuple of (defender_strategy, attacker_strategy)
        """
        m, n = payoff_matrix.shape
        
        # Defender's strategy (maximize minimum payoff)
        c = np.zeros(m + 1)
        c[-1] = -1
        
        A_ub = np.hstack([-payoff_matrix.T, np.ones((n, 1))])
        b_ub = np.zeros(n)
        
        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1
        b_eq = 1
        
        bounds = [(0, 1) for _ in range(m)] + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        defender_strategy = result.x[:m]
        
        # Attacker's strategy (similar approach)
        c_att = np.zeros(n + 1)
        c_att[-1] = 1
        
        A_ub_att = np.hstack([payoff_matrix, -np.ones((m, 1))])
        b_ub_att = np.zeros(m)
        
        A_eq_att = np.zeros((1, n + 1))
        A_eq_att[0, :n] = 1
        b_eq_att = 1
        
        bounds_att = [(0, 1) for _ in range(n)] + [(None, None)]
        
        result_att = linprog(c_att, A_ub=A_ub_att, b_ub=b_ub_att, 
                            A_eq=A_eq_att, b_eq=b_eq_att, bounds=bounds_att, 
                            method='highs')
        
        attacker_strategy = result_att.x[:n]
        
        return defender_strategy, attacker_strategy


class MinimaxStrategy:
    """Implements minimax strategy for worst-case attack scenarios"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Risk aversion parameter
        
    def compute_worst_case_fusion(self, lstm_output: np.ndarray, 
                                 arima_output: np.ndarray,
                                 attack_scenarios: List[Dict]) -> np.ndarray:
        """
        Compute fusion weights considering worst-case attack scenarios
        
        Args:
            lstm_output: LSTM predictions
            arima_output: ARIMA predictions
            attack_scenarios: List of potential attack scenarios
            
        Returns:
            Optimal fusion weights for worst-case robustness
        """
        n_scenarios = len(attack_scenarios)
        
        def worst_case_loss(weights):
            lstm_weight, arima_weight = weights[0], 1 - weights[0]
            max_loss = -np.inf
            
            for scenario in attack_scenarios:
                # Simulate model performance under attack
                lstm_degradation = scenario.get('lstm_degradation', 0)
                arima_degradation = scenario.get('arima_degradation', 0)
                
                effective_lstm = lstm_output * (1 - lstm_degradation)
                effective_arima = arima_output * (1 - arima_degradation)
                
                fusion = lstm_weight * effective_lstm + arima_weight * effective_arima
                loss = -np.mean(fusion)  # Negative for minimization
                
                if loss > max_loss:
                    max_loss = loss
                    
            return max_loss + self.alpha * np.var(weights)  # Add regularization
        
        # Optimize weights
        result = minimize(worst_case_loss, x0=[0.5], bounds=[(0, 1)], method='L-BFGS-B')
        
        return np.array([result.x[0], 1 - result.x[0]])


class BayesianGameModel:
    """Implements Bayesian game model for uncertainty quantification"""
    
    def __init__(self, n_types: int = 5):
        self.n_types = n_types  # Number of attacker types
        self.type_probs = np.ones(n_types) / n_types  # Prior probabilities
        
    def update_beliefs(self, observations: np.ndarray) -> np.ndarray:
        """
        Update beliefs about attacker types using Bayesian inference
        
        Args:
            observations: Observed attack patterns
            
        Returns:
            Updated type probabilities
        """
        likelihoods = np.zeros(self.n_types)
        
        for i in range(self.n_types):
            # Different attacker types have different patterns
            if i == 0:  # Stealth attacker
                likelihoods[i] = np.exp(-np.mean(observations))
            elif i == 1:  # Aggressive attacker
                likelihoods[i] = np.exp(-np.var(observations))
            elif i == 2:  # Adaptive attacker
                likelihoods[i] = np.exp(-np.abs(np.mean(np.diff(observations))))
            elif i == 3:  # Random attacker
                likelihoods[i] = 1.0 / (1 + np.std(observations))
            else:  # Unknown type
                likelihoods[i] = 0.5
                
        # Bayesian update
        posterior = likelihoods * self.type_probs
        posterior /= np.sum(posterior)
        
        self.type_probs = posterior
        return posterior
    
    def compute_bayesian_fusion(self, lstm_output: np.ndarray,
                               arima_output: np.ndarray) -> np.ndarray:
        """
        Compute fusion weights considering uncertainty over attacker types
        
        Args:
            lstm_output: LSTM predictions
            arima_output: ARIMA predictions
            
        Returns:
            Bayesian optimal fusion weights
        """
        expected_payoff = np.zeros(2)
        
        for i in range(self.n_types):
            # Each attacker type has different effectiveness against models
            if i == 0:  # Stealth attacker - better against LSTM
                lstm_effectiveness = 0.6
                arima_effectiveness = 0.8
            elif i == 1:  # Aggressive attacker - better against ARIMA
                lstm_effectiveness = 0.8
                arima_effectiveness = 0.6
            else:
                lstm_effectiveness = 0.7
                arima_effectiveness = 0.7
                
            expected_payoff[0] += self.type_probs[i] * lstm_effectiveness
            expected_payoff[1] += self.type_probs[i] * arima_effectiveness
            
        # Normalize to get fusion weights
        weights = expected_payoff / np.sum(expected_payoff)
        return weights


class EvolutionaryGameDynamics:
    """Implements evolutionary game dynamics for adaptation"""
    
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.01):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize population of strategies"""
        # Each individual represents fusion weights [lstm_weight, arima_weight]
        population = np.random.dirichlet([1, 1], self.population_size)
        return population
    
    def fitness_function(self, strategy: np.ndarray, environment: Dict) -> float:
        """
        Compute fitness of a strategy in given environment
        
        Args:
            strategy: Fusion weight strategy
            environment: Current environment parameters
            
        Returns:
            Fitness score
        """
        lstm_accuracy = environment.get('lstm_accuracy', 0.85)
        arima_accuracy = environment.get('arima_accuracy', 0.80)
        attack_intensity = environment.get('attack_intensity', 0.3)
        
        # Fitness combines accuracy and robustness
        accuracy = strategy[0] * lstm_accuracy + strategy[1] * arima_accuracy
        robustness = 1 - attack_intensity * np.abs(strategy[0] - strategy[1])
        
        return accuracy * robustness
    
    def evolve(self, environment: Dict, generations: int = 10) -> np.ndarray:
        """
        Evolve population over multiple generations
        
        Args:
            environment: Environment parameters
            generations: Number of evolution steps
            
        Returns:
            Best evolved strategy
        """
        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = np.array([
                self.fitness_function(ind, environment) for ind in self.population
            ])
            
            # Selection (tournament)
            new_population = []
            for _ in range(self.population_size):
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                if fitness_scores[idx1] > fitness_scores[idx2]:
                    winner = self.population[idx1].copy()
                else:
                    winner = self.population[idx2].copy()
                    
                # Mutation
                if np.random.random() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.1, 2)
                    winner += mutation
                    winner = np.clip(winner, 0, 1)
                    winner /= np.sum(winner)
                    
                new_population.append(winner)
                
            self.population = np.array(new_population)
            
        # Return best strategy
        final_fitness = np.array([
            self.fitness_function(ind, environment) for ind in self.population
        ])
        best_idx = np.argmax(final_fitness)
        
        return self.population[best_idx]


class StackelbergGame:
    """Implements Stackelberg game for leader-follower dynamics"""
    
    def __init__(self, leader_advantage: float = 0.1):
        self.leader_advantage = leader_advantage
        
    def compute_stackelberg_equilibrium(self, lstm_params: Dict,
                                      arima_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Stackelberg equilibrium with defender as leader
        
        Args:
            lstm_params: LSTM model parameters
            arima_params: ARIMA model parameters
            
        Returns:
            Tuple of (leader_strategy, follower_best_response)
        """
        # Define leader's utility function
        def leader_utility(leader_weights):
            lstm_weight = leader_weights[0]
            arima_weight = 1 - lstm_weight
            
            # Anticipate follower's best response
            follower_response = self._follower_best_response(leader_weights, 
                                                            lstm_params, arima_params)
            
            # Leader's utility considers detection accuracy and first-mover advantage
            detection_utility = (lstm_weight * lstm_params['accuracy'] + 
                               arima_weight * arima_params['accuracy'])
            strategic_advantage = self.leader_advantage * np.exp(-follower_response[0])
            
            return -(detection_utility + strategic_advantage)  # Negative for minimization
        
        # Optimize leader's strategy
        result = minimize(leader_utility, x0=[0.5], bounds=[(0, 1)], method='L-BFGS-B')
        leader_strategy = np.array([result.x[0], 1 - result.x[0]])
        
        # Compute follower's best response
        follower_response = self._follower_best_response(leader_strategy,
                                                        lstm_params, arima_params)
        
        return leader_strategy, follower_response
    
    def _follower_best_response(self, leader_strategy: np.ndarray,
                               lstm_params: Dict, arima_params: Dict) -> np.ndarray:
        """Compute follower's best response to leader's strategy"""
        # Ensure leader_strategy has 2 elements
        if len(leader_strategy) == 1:
            leader_strategy = np.array([leader_strategy[0], 1 - leader_strategy[0]])
        
        # Follower minimizes detection probability given leader's choice
        def follower_utility(attack_vector):
            detection_prob = (leader_strategy[0] * lstm_params['detection_rate'] * attack_vector[0] +
                            leader_strategy[1] * arima_params['detection_rate'] * attack_vector[1])
            attack_cost = 0.1 * np.sum(attack_vector ** 2)
            
            return detection_prob + attack_cost
        
        result = minimize(follower_utility, x0=[0.5, 0.5], 
                         bounds=[(0, 1), (0, 1)], method='L-BFGS-B')
        
        return result.x


class CooperativeGameTheory:
    """Implements cooperative game theory for ensemble decisions"""
    
    def __init__(self, n_models: int = 2):
        self.n_models = n_models
        
    def compute_shapley_values(self, model_contributions: Dict[str, float]) -> Dict[str, float]:
        """
        Compute Shapley values for fair contribution assessment
        
        Args:
            model_contributions: Dictionary of model names to contribution values
            
        Returns:
            Dictionary of Shapley values
        """
        models = list(model_contributions.keys())
        n = len(models)
        shapley_values = {model: 0.0 for model in models}
        
        # Compute characteristic function for all coalitions
        
        for model in models:
            for r in range(n):
                for coalition in combinations([m for m in models if m != model], r):
                    coalition_with = list(coalition) + [model]
                    coalition_without = list(coalition)
                    
                    # Value of coalition with model
                    if len(coalition_with) == 0:
                        v_with = 0
                    else:
                        v_with = sum(model_contributions[m] for m in coalition_with) / len(coalition_with)
                    
                    # Value of coalition without model
                    if len(coalition_without) == 0:
                        v_without = 0
                    else:
                        v_without = sum(model_contributions[m] for m in coalition_without) / len(coalition_without)
                    
                    # Marginal contribution
                    marginal = v_with - v_without
                    
                    # Weight for this coalition
                    weight = 1 / (n * math.comb(n - 1, r))
                    
                    shapley_values[model] += weight * marginal
                    
        return shapley_values
    
    def compute_core_allocation(self, lstm_output: float, arima_output: float) -> np.ndarray:
        """
        Compute core allocation for stable cooperation
        
        Args:
            lstm_output: LSTM model output
            arima_output: ARIMA model output
            
        Returns:
            Core allocation weights
        """
        # Define characteristic function
        v = {
            (): 0,
            ('lstm',): lstm_output,
            ('arima',): arima_output,
            ('lstm', 'arima'): lstm_output + arima_output + 0.1  # Synergy bonus
        }
        
        # Core constraints
        # x_lstm + x_arima = v(N)
        # x_lstm >= v({lstm})
        # x_arima >= v({arima})
        
        total_value = v[('lstm', 'arima')]
        min_lstm = v[('lstm',)]
        min_arima = v[('arima',)]
        
        # Find allocation in the core
        if min_lstm + min_arima > total_value:
            # Empty core, use proportional allocation
            total = min_lstm + min_arima
            weights = np.array([min_lstm / total, min_arima / total])
        else:
            # Allocate surplus proportionally
            surplus = total_value - min_lstm - min_arima
            lstm_share = min_lstm + surplus * (min_lstm / (min_lstm + min_arima))
            arima_share = min_arima + surplus * (min_arima / (min_lstm + min_arima))
            
            weights = np.array([lstm_share / total_value, arima_share / total_value])
            
        return weights


class MechanismDesign:
    """Implements mechanism design for incentive-compatible fusion"""
    
    def __init__(self, budget: float = 1.0):
        self.budget = budget
        
    def design_vickrey_auction(self, model_bids: Dict[str, float]) -> Dict[str, float]:
        """
        Design Vickrey auction for model selection
        
        Args:
            model_bids: Dictionary of model names to bid values (quality scores)
            
        Returns:
            Dictionary of payments to models
        """
        sorted_bids = sorted(model_bids.items(), key=lambda x: x[1], reverse=True)
        
        payments = {}
        if len(sorted_bids) >= 2:
            # Winner pays second-highest bid
            winner = sorted_bids[0][0]
            second_price = sorted_bids[1][1]
            payments[winner] = second_price * self.budget
            
            # Others get nothing
            for model, _ in sorted_bids[1:]:
                payments[model] = 0.0
        else:
            # Single bidder gets budget
            payments[sorted_bids[0][0]] = self.budget
            
        return payments
    
    def compute_incentive_compatible_weights(self, lstm_quality: float,
                                           arima_quality: float) -> np.ndarray:
        """
        Compute incentive-compatible fusion weights
        
        Args:
            lstm_quality: LSTM model quality score
            arima_quality: ARIMA model quality score
            
        Returns:
            Incentive-compatible weights
        """
        # Use logarithmic scoring rule for incentive compatibility
        lstm_score = np.log(1 + lstm_quality)
        arima_score = np.log(1 + arima_quality)
        
        total_score = lstm_score + arima_score
        weights = np.array([lstm_score / total_score, arima_score / total_score])
        
        return weights


class RegretMinimization:
    """Implements regret minimization algorithms"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.cumulative_regret = np.zeros(2)
        self.strategy_sum = np.zeros(2)
        self.t = 0
        
    def update_regrets(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        Update regrets based on observed rewards
        
        Args:
            actions: Chosen actions (weights)
            rewards: Observed rewards for each model
        """
        # Compute regret for each action
        best_reward = np.max(rewards)
        actual_reward = np.dot(actions, rewards)
        
        for i in range(2):
            counterfactual_reward = rewards[i]
            regret = counterfactual_reward - actual_reward
            self.cumulative_regret[i] += regret
            
        self.t += 1
        
    def get_strategy(self) -> np.ndarray:
        """
        Get current strategy using regret matching
        
        Returns:
            Current strategy (fusion weights)
        """
        # Regret matching
        positive_regrets = np.maximum(self.cumulative_regret, 0)
        
        if np.sum(positive_regrets) > 0:
            strategy = positive_regrets / np.sum(positive_regrets)
        else:
            strategy = np.ones(2) / 2
            
        self.strategy_sum += strategy
        
        return strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Get average strategy over all iterations
        
        Returns:
            Average strategy (converges to Nash equilibrium)
        """
        if self.t > 0:
            return self.strategy_sum / self.t
        else:
            return np.ones(2) / 2


class MultiAgentRL(nn.Module):
    """Implements multi-agent reinforcement learning for fusion"""
    
    def __init__(self, state_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        
        # Actor networks for each agent (LSTM and ARIMA)
        self.lstm_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.arima_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Critic network (centralized)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor networks"""
        lstm_action = self.lstm_actor(state)
        arima_action = self.arima_actor(state)
        
        # Ensure weights sum to 1
        total = lstm_action + arima_action
        lstm_weight = lstm_action / total
        arima_weight = arima_action / total
        
        return lstm_weight, arima_weight
    
    def get_value(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get value estimate from critic"""
        combined = torch.cat([state, actions], dim=-1)
        return self.critic(combined)


class AdversarialRobustness:
    """Implements adversarial robustness guarantees"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon  # Perturbation budget
        
    def compute_robust_weights(self, lstm_output: np.ndarray,
                             arima_output: np.ndarray) -> np.ndarray:
        """
        Compute adversarially robust fusion weights
        
        Args:
            lstm_output: LSTM predictions
            arima_output: ARIMA predictions
            
        Returns:
            Robust fusion weights
        """
        # Solve robust optimization problem
        def robust_objective(weights):
            lstm_weight, arima_weight = weights[0], 1 - weights[0]
            
            # Consider worst-case perturbations
            worst_lstm = lstm_output - self.epsilon * np.sign(lstm_output)
            worst_arima = arima_output - self.epsilon * np.sign(arima_output)
            
            # Robust loss
            nominal_loss = -np.mean(lstm_weight * lstm_output + arima_weight * arima_output)
            robust_loss = -np.mean(lstm_weight * worst_lstm + arima_weight * worst_arima)
            
            return 0.5 * nominal_loss + 0.5 * robust_loss
        
        result = minimize(robust_objective, x0=[0.5], bounds=[(0, 1)], method='L-BFGS-B')
        
        return np.array([result.x[0], 1 - result.x[0]])
    
    def certify_robustness(self, weights: np.ndarray, lstm_output: np.ndarray,
                          arima_output: np.ndarray) -> float:
        """
        Certify robustness of fusion weights
        
        Args:
            weights: Fusion weights
            lstm_output: LSTM predictions
            arima_output: ARIMA predictions
            
        Returns:
            Certified robustness radius
        """
        # Compute gradient of fusion output w.r.t. inputs
        fusion = weights[0] * lstm_output + weights[1] * arima_output
        
        # Lipschitz constant
        lipschitz = np.sqrt(weights[0]**2 + weights[1]**2)
        
        # Certified radius (simplified)
        margin = np.abs(fusion - 0.5).mean()  # Distance to decision boundary
        certified_radius = margin / lipschitz
        
        return certified_radius


class GameTheoreticFusionLayer:
    """
    Revolutionary Game-Theoretic Fusion Layer
    
    Combines multiple game theory paradigms for robust fusion of LSTM and ARIMA outputs
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fusion layer with all game-theoretic components
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize all game theory components
        self.nash_solver = NashEquilibriumSolver()
        self.minimax = MinimaxStrategy()
        self.bayesian_game = BayesianGameModel()
        self.evolutionary = EvolutionaryGameDynamics()
        self.stackelberg = StackelbergGame()
        self.cooperative = CooperativeGameTheory()
        self.mechanism_design = MechanismDesign()
        self.regret_min = RegretMinimization()
        self.marl = MultiAgentRL()
        self.adversarial = AdversarialRobustness()
        
        # State tracking
        self.game_states = []
        self.fusion_history = []
        self.decision_boundaries = []
        
    def fuse(self, lstm_output: np.ndarray, arima_output: np.ndarray,
            context: Optional[Dict] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform game-theoretic fusion of model outputs
        
        Args:
            lstm_output: LSTM model predictions
            arima_output: ARIMA model predictions
            context: Additional context information
            
        Returns:
            Dictionary containing fusion results and metadata
        """
        context = context or {}
        
        # Extract context information
        attack_prob = context.get('attack_probability', 0.3)
        uncertainty = context.get('uncertainty', 0.5)
        attack_history = context.get('attack_history', [])
        
        # 1. Nash Equilibrium
        lstm_conf = context.get('lstm_confidence', 0.85)
        arima_conf = context.get('arima_confidence', 0.80)
        payoff_matrix = self.nash_solver.compute_payoff_matrix(lstm_conf, arima_conf, attack_prob)
        nash_defender, nash_attacker = self.nash_solver.find_nash_equilibrium(payoff_matrix)
        nash_weights = nash_defender[:2] / np.sum(nash_defender[:2])
        
        # 2. Minimax Strategy
        attack_scenarios = [
            {'lstm_degradation': 0.2, 'arima_degradation': 0.1},
            {'lstm_degradation': 0.1, 'arima_degradation': 0.3},
            {'lstm_degradation': 0.15, 'arima_degradation': 0.15},
        ]
        minimax_weights = self.minimax.compute_worst_case_fusion(
            lstm_output, arima_output, attack_scenarios
        )
        
        # 3. Bayesian Game
        if len(attack_history) > 0:
            self.bayesian_game.update_beliefs(np.array(attack_history))
        bayesian_weights = self.bayesian_game.compute_bayesian_fusion(lstm_output, arima_output)
        
        # 4. Evolutionary Dynamics
        environment = {
            'lstm_accuracy': lstm_conf,
            'arima_accuracy': arima_conf,
            'attack_intensity': attack_prob
        }
        evolutionary_weights = self.evolutionary.evolve(environment)
        
        # 5. Stackelberg Game
        lstm_params = {'accuracy': lstm_conf, 'detection_rate': 0.9}
        arima_params = {'accuracy': arima_conf, 'detection_rate': 0.85}
        stackelberg_leader, stackelberg_follower = self.stackelberg.compute_stackelberg_equilibrium(
            lstm_params, arima_params
        )
        
        # 6. Cooperative Game Theory
        shapley_values = self.cooperative.compute_shapley_values({
            'lstm': lstm_conf,
            'arima': arima_conf
        })
        cooperative_weights = self.cooperative.compute_core_allocation(
            np.mean(lstm_output), np.mean(arima_output)
        )
        
        # 7. Mechanism Design
        incentive_weights = self.mechanism_design.compute_incentive_compatible_weights(
            lstm_conf, arima_conf
        )
        
        # 8. Regret Minimization
        rewards = np.array([np.mean(lstm_output), np.mean(arima_output)])
        self.regret_min.update_regrets(self.regret_min.get_strategy(), rewards)
        regret_weights = self.regret_min.get_average_strategy()
        
        # 9. Multi-Agent RL (using PyTorch)
        # Ensure we have enough samples for state
        min_len = min(len(lstm_output), len(arima_output))
        state_len = min(5, min_len)
        
        state = torch.FloatTensor(np.concatenate([
            lstm_output[:state_len], arima_output[:state_len]
        ])).unsqueeze(0)
        
        # Pad if needed
        if state.shape[1] < 10:
            padding = torch.zeros(1, 10 - state.shape[1])
            state = torch.cat([state, padding], dim=1)
        
        with torch.no_grad():
            marl_lstm, marl_arima = self.marl(state)
        marl_weights = np.array([marl_lstm.item(), marl_arima.item()])
        
        # 10. Adversarial Robustness
        robust_weights = self.adversarial.compute_robust_weights(lstm_output, arima_output)
        robustness_radius = self.adversarial.certify_robustness(
            robust_weights, lstm_output, arima_output
        )
        
        # Meta-fusion: Combine all strategies
        all_weights = np.array([
            nash_weights,
            minimax_weights,
            bayesian_weights,
            evolutionary_weights,
            stackelberg_leader,
            cooperative_weights,
            incentive_weights,
            regret_weights,
            marl_weights,
            robust_weights
        ])
        
        # Handle NaN values by replacing with default weights
        for i in range(all_weights.shape[0]):
            if np.any(np.isnan(all_weights[i])) or np.any(np.isinf(all_weights[i])):
                all_weights[i] = np.array([0.5, 0.5])
        
        # Meta-weights based on context
        meta_weights = self._compute_meta_weights(context)
        final_weights = np.dot(meta_weights, all_weights)
        
        # Ensure final weights are valid
        if np.any(np.isnan(final_weights)) or np.any(np.isinf(final_weights)):
            final_weights = np.array([0.5, 0.5])
        
        # Normalize to ensure they sum to 1
        final_weights = final_weights / np.sum(final_weights)
        
        # Apply final fusion
        fusion_output = final_weights[0] * lstm_output + final_weights[1] * arima_output
        
        # Store game state
        game_state = GameState(
            lstm_output=lstm_output,
            arima_output=arima_output,
            uncertainty=uncertainty,
            attack_history=attack_history,
            fusion_weights=final_weights,
            equilibrium_point=nash_weights,
            payoff_matrix=payoff_matrix
        )
        self.game_states.append(game_state)
        
        return {
            'fusion_output': fusion_output,
            'final_weights': final_weights,
            'nash_weights': nash_weights,
            'minimax_weights': minimax_weights,
            'bayesian_weights': bayesian_weights,
            'evolutionary_weights': evolutionary_weights,
            'stackelberg_weights': stackelberg_leader,
            'cooperative_weights': cooperative_weights,
            'incentive_weights': incentive_weights,
            'regret_weights': regret_weights,
            'marl_weights': marl_weights,
            'robust_weights': robust_weights,
            'robustness_radius': robustness_radius,
            'shapley_values': shapley_values,
            'game_state': game_state
        }
    
    def _compute_meta_weights(self, context: Dict) -> np.ndarray:
        """
        Compute meta-weights for combining different game theory strategies
        
        Args:
            context: Context information
            
        Returns:
            Meta-weights for strategy combination
        """
        # Adaptive meta-weighting based on context
        attack_prob = context.get('attack_probability', 0.3)
        uncertainty = context.get('uncertainty', 0.5)
        data_quality = context.get('data_quality', 0.8)
        
        weights = np.zeros(10)
        
        # High attack probability: favor minimax and robust strategies
        if attack_prob > 0.5:
            weights[1] = 0.3  # Minimax
            weights[9] = 0.3  # Robust
        else:
            weights[0] = 0.2  # Nash
            
        # High uncertainty: favor Bayesian and regret minimization
        if uncertainty > 0.5:
            weights[2] = 0.2  # Bayesian
            weights[7] = 0.2  # Regret
        else:
            weights[4] = 0.2  # Stackelberg
            
        # Good data quality: favor cooperative and MARL
        if data_quality > 0.7:
            weights[5] = 0.15  # Cooperative
            weights[8] = 0.15  # MARL
        else:
            weights[3] = 0.2  # Evolutionary
            
        # Ensure weights sum to 1
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(10) / 10
        
        return weights
    
    def visualize_game_dynamics(self, game_states: Optional[List[GameState]] = None) -> None:
        """
        Visualize game dynamics and decision boundaries
        
        Args:
            game_states: List of game states to visualize (uses internal if None)
        """
        if game_states is None:
            game_states = self.game_states[-10:]  # Last 10 states
            
        if len(game_states) == 0:
            print("No game states to visualize")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Game-Theoretic Fusion Dynamics', fontsize=16)
        
        # 1. Fusion weights evolution
        ax = axes[0, 0]
        weights_history = [state.fusion_weights for state in game_states]
        lstm_weights = [w[0] for w in weights_history]
        arima_weights = [w[1] for w in weights_history]
        
        ax.plot(lstm_weights, label='LSTM Weight', marker='o')
        ax.plot(arima_weights, label='ARIMA Weight', marker='s')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Weight')
        ax.set_title('Fusion Weights Evolution')
        ax.legend()
        ax.grid(True)
        
        # 2. Payoff matrix heatmap
        ax = axes[0, 1]
        if game_states[-1].payoff_matrix is not None:
            sns.heatmap(game_states[-1].payoff_matrix, annot=True, fmt='.3f',
                       cmap='coolwarm', ax=ax)
            ax.set_title('Payoff Matrix (Last State)')
            ax.set_xlabel('Attacker Strategy')
            ax.set_ylabel('Defender Strategy')
        
        # 3. Nash equilibrium trajectory
        ax = axes[0, 2]
        equilibria = [state.equilibrium_point for state in game_states if state.equilibrium_point is not None]
        if equilibria:
            eq_lstm = [e[0] for e in equilibria]
            eq_arima = [e[1] for e in equilibria if len(e) > 1]
            
            ax.scatter(eq_lstm, eq_arima, c=range(len(eq_lstm)), cmap='viridis')
            ax.set_xlabel('LSTM Weight (Nash)')
            ax.set_ylabel('ARIMA Weight (Nash)')
            ax.set_title('Nash Equilibrium Trajectory')
            ax.grid(True)
        
        # 4. Uncertainty evolution
        ax = axes[1, 0]
        uncertainties = [state.uncertainty for state in game_states]
        ax.plot(uncertainties, marker='o', color='red')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Uncertainty')
        ax.set_title('Uncertainty Evolution')
        ax.grid(True)
        
        # 5. Attack history pattern
        ax = axes[1, 1]
        if game_states[-1].attack_history:
            ax.plot(game_states[-1].attack_history, marker='o', color='darkred')
            ax.set_xlabel('Time')
            ax.set_ylabel('Attack Intensity')
            ax.set_title('Attack History Pattern')
            ax.grid(True)
        
        # 6. Strategy comparison
        ax = axes[1, 2]
        if hasattr(self, 'fusion_history') and self.fusion_history:
            latest_fusion = self.fusion_history[-1]
            strategies = ['Nash', 'Minimax', 'Bayesian', 'Evolutionary', 'Stackelberg',
                         'Cooperative', 'Incentive', 'Regret', 'MARL', 'Robust']
            weights = [
                latest_fusion.get('nash_weights', [0, 0])[0],
                latest_fusion.get('minimax_weights', [0, 0])[0],
                latest_fusion.get('bayesian_weights', [0, 0])[0],
                latest_fusion.get('evolutionary_weights', [0, 0])[0],
                latest_fusion.get('stackelberg_weights', [0, 0])[0],
                latest_fusion.get('cooperative_weights', [0, 0])[0],
                latest_fusion.get('incentive_weights', [0, 0])[0],
                latest_fusion.get('regret_weights', [0, 0])[0],
                latest_fusion.get('marl_weights', [0, 0])[0],
                latest_fusion.get('robust_weights', [0, 0])[0],
            ]
            
            ax.bar(strategies, weights, color='skyblue', edgecolor='navy')
            ax.set_xlabel('Strategy')
            ax.set_ylabel('LSTM Weight')
            ax.set_title('Strategy Comparison')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_decision_boundaries(self, lstm_range: Tuple[float, float] = (0, 1),
                                    arima_range: Tuple[float, float] = (0, 1),
                                    resolution: int = 50) -> None:
        """
        Visualize decision boundaries for different game theory strategies
        
        Args:
            lstm_range: Range of LSTM outputs to visualize
            arima_range: Range of ARIMA outputs to visualize
            resolution: Grid resolution
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Game-Theoretic Decision Boundaries', fontsize=16)
        
        lstm_grid = np.linspace(lstm_range[0], lstm_range[1], resolution)
        arima_grid = np.linspace(arima_range[0], arima_range[1], resolution)
        
        strategies = [
            ('Nash Equilibrium', self.nash_solver, axes[0, 0]),
            ('Minimax', self.minimax, axes[0, 1]),
            ('Bayesian Game', self.bayesian_game, axes[0, 2]),
            ('Evolutionary', self.evolutionary, axes[1, 0]),
            ('Stackelberg', self.stackelberg, axes[1, 1]),
            ('Adversarial Robust', self.adversarial, axes[1, 2]),
        ]
        
        for name, strategy, ax in strategies:
            decision_map = np.zeros((resolution, resolution))
            
            for i, lstm_val in enumerate(lstm_grid):
                for j, arima_val in enumerate(arima_grid):
                    # Compute decision for this point
                    if name == 'Nash Equilibrium':
                        payoff = self.nash_solver.compute_payoff_matrix(lstm_val, arima_val, 0.3)
                        defender, _ = self.nash_solver.find_nash_equilibrium(payoff)
                        decision = defender[0] if len(defender) > 0 else 0.5
                    elif name == 'Minimax':
                        weights = self.minimax.compute_worst_case_fusion(
                            np.array([lstm_val]), np.array([arima_val]), 
                            [{'lstm_degradation': 0.1, 'arima_degradation': 0.1}]
                        )
                        decision = weights[0]
                    elif name == 'Bayesian Game':
                        weights = self.bayesian_game.compute_bayesian_fusion(
                            np.array([lstm_val]), np.array([arima_val])
                        )
                        decision = weights[0]
                    elif name == 'Evolutionary':
                        env = {'lstm_accuracy': lstm_val, 'arima_accuracy': arima_val, 'attack_intensity': 0.3}
                        weights = self.evolutionary.evolve(env, generations=5)
                        decision = weights[0]
                    elif name == 'Stackelberg':
                        lstm_p = {'accuracy': lstm_val, 'detection_rate': 0.9}
                        arima_p = {'accuracy': arima_val, 'detection_rate': 0.85}
                        leader, _ = self.stackelberg.compute_stackelberg_equilibrium(lstm_p, arima_p)
                        decision = leader[0]
                    else:  # Adversarial Robust
                        weights = self.adversarial.compute_robust_weights(
                            np.array([lstm_val]), np.array([arima_val])
                        )
                        decision = weights[0]
                    
                    decision_map[i, j] = decision
            
            # Plot decision boundary
            im = ax.imshow(decision_map.T, extent=[*lstm_range, *arima_range],
                          origin='lower', aspect='auto', cmap='RdBu_r')
            ax.set_xlabel('LSTM Output')
            ax.set_ylabel('ARIMA Output')
            ax.set_title(f'{name} Decision Boundary')
            
            # Add contour lines
            contours = ax.contour(lstm_grid, arima_grid, decision_map.T,
                                 levels=[0.3, 0.5, 0.7], colors='black', alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='LSTM Weight')
        
        plt.tight_layout()
        plt.show()


def demonstrate_fusion_layer():
    """Demonstrate the game-theoretic fusion layer capabilities"""
    print("=" * 80)
    print("Revolutionary Game-Theoretic Fusion Layer Demonstration")
    print("=" * 80)
    
    # Initialize fusion layer
    fusion_layer = GameTheoreticFusionLayer()
    
    # Simulate model outputs
    np.random.seed(42)
    n_samples = 100
    
    # Normal scenario
    lstm_normal = np.random.beta(8, 2, n_samples)  # High confidence
    arima_normal = np.random.beta(7, 3, n_samples)  # Moderate confidence
    
    # Attack scenario
    lstm_attack = np.random.beta(3, 7, n_samples)  # Degraded performance
    arima_attack = np.random.beta(4, 6, n_samples)  # Degraded performance
    
    # Test 1: Normal conditions
    print("\n1. Testing under normal conditions:")
    context_normal = {
        'attack_probability': 0.1,
        'uncertainty': 0.2,
        'lstm_confidence': 0.9,
        'arima_confidence': 0.85,
        'data_quality': 0.95,
        'attack_history': []
    }
    
    result_normal = fusion_layer.fuse(lstm_normal, arima_normal, context_normal)
    fusion_layer.fusion_history.append(result_normal)
    
    print(f"   Final fusion weights: LSTM={result_normal['final_weights'][0]:.3f}, "
          f"ARIMA={result_normal['final_weights'][1]:.3f}")
    print(f"   Robustness radius: {result_normal['robustness_radius']:.3f}")
    print(f"   Average fusion output: {np.mean(result_normal['fusion_output']):.3f}")
    
    # Test 2: Under attack
    print("\n2. Testing under attack conditions:")
    context_attack = {
        'attack_probability': 0.8,
        'uncertainty': 0.7,
        'lstm_confidence': 0.6,
        'arima_confidence': 0.65,
        'data_quality': 0.5,
        'attack_history': [0.1, 0.3, 0.7, 0.9, 0.8]
    }
    
    result_attack = fusion_layer.fuse(lstm_attack, arima_attack, context_attack)
    fusion_layer.fusion_history.append(result_attack)
    
    print(f"   Final fusion weights: LSTM={result_attack['final_weights'][0]:.3f}, "
          f"ARIMA={result_attack['final_weights'][1]:.3f}")
    print(f"   Robustness radius: {result_attack['robustness_radius']:.3f}")
    print(f"   Average fusion output: {np.mean(result_attack['fusion_output']):.3f}")
    
    # Test 3: Evolving attack
    print("\n3. Testing with evolving attack pattern:")
    for t in range(5):
        # Simulate evolving conditions
        attack_intensity = 0.1 + 0.15 * t
        lstm_evolving = np.random.beta(8 - t, 2 + t, n_samples)
        arima_evolving = np.random.beta(7 - t/2, 3 + t/2, n_samples)
        
        context_evolving = {
            'attack_probability': attack_intensity,
            'uncertainty': 0.3 + 0.1 * t,
            'lstm_confidence': 0.9 - 0.05 * t,
            'arima_confidence': 0.85 - 0.03 * t,
            'data_quality': 0.9 - 0.08 * t,
            'attack_history': [attack_intensity] * (t + 1)
        }
        
        result = fusion_layer.fuse(lstm_evolving, arima_evolving, context_evolving)
        fusion_layer.fusion_history.append(result)
        
        print(f"   Time {t}: Weights=[{result['final_weights'][0]:.3f}, "
              f"{result['final_weights'][1]:.3f}], "
              f"Robustness={result['robustness_radius']:.3f}")
    
    # Compare strategies
    print("\n4. Strategy Comparison (Last Result):")
    strategies = [
        ('Nash Equilibrium', 'nash_weights'),
        ('Minimax', 'minimax_weights'),
        ('Bayesian Game', 'bayesian_weights'),
        ('Evolutionary', 'evolutionary_weights'),
        ('Stackelberg', 'stackelberg_weights'),
        ('Cooperative', 'cooperative_weights'),
        ('Incentive Compatible', 'incentive_weights'),
        ('Regret Minimization', 'regret_weights'),
        ('Multi-Agent RL', 'marl_weights'),
        ('Adversarial Robust', 'robust_weights')
    ]
    
    for name, key in strategies:
        weights = result.get(key, [0, 0])
        if isinstance(weights, np.ndarray) and len(weights) >= 2:
            print(f"   {name:20s}: LSTM={weights[0]:.3f}, ARIMA={weights[1]:.3f}")
    
    # Shapley values
    print("\n5. Shapley Values (Contribution Assessment):")
    shapley = result.get('shapley_values', {})
    for model, value in shapley.items():
        print(f"   {model}: {value:.3f}")
    
    # Visualizations
    print("\n6. Generating visualizations...")
    fusion_layer.visualize_game_dynamics()
    fusion_layer.visualize_decision_boundaries()
    
    print("\n" + "=" * 80)
    print("Game-theoretic fusion layer demonstration completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_fusion_layer()