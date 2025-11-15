"""
WB_logic.py - RL Training Logic with Value-Based and Gradient-Based Methods
Contains classes and functions for various RL algorithms.
"""

import os
# Suppress TensorFlow oneDNN custom operations info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import gymnasium as gym
from collections import deque
import random
from typing import Tuple, List, Optional, Dict, Any


class ReplayBuffer:
    """Experience replay buffer for off-policy methods."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using sum tree for efficient sampling."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Assign maximum priority to new experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch with priorities and compute importance sampling weights.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            return None
        
        # Get valid priorities
        priorities = self.priorities[:buffer_len]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(buffer_len, size=min(batch_size, buffer_len), p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (buffer_len * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent (Value-Based)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary with hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update = config.get('target_update', 10)
        self.hidden_layers = config.get('hidden_layers', [128, 128])
        
        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Replay buffer
        self.memory = ReplayBuffer(config.get('memory_size', 10000))
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Training metrics
        self.episode_count = 0
        self.current_reward = 0
        self.current_epsilon = self.epsilon
    
    def _build_network(self) -> Model:
        """Build Q-network."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        outputs = layers.Dense(self.action_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_network(np.expand_dims(state, 0), training=False)
            return np.argmax(q_values[0])
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute target Q-values
        next_q_values = self.target_network(next_states, training=False)
        target_q_values = rewards + (1 - dones) * self.gamma * np.max(next_q_values, axis=1)
        
        # Train network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss.numpy()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.current_epsilon = self.epsilon


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent (Value-Based)."""
    
    def train_step(self):
        """Perform one training step with Double DQN update."""
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Double DQN: use policy network to select actions, target network to evaluate
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1, output_type=tf.int32)
        next_q_values = self.target_network(next_states, training=False)
        next_q_values = tf.gather_nd(next_q_values, 
                                     tf.stack([tf.range(self.batch_size, dtype=tf.int32), next_actions], axis=1))
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Train network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss.numpy()


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN Agent (Value-Based)."""
    
    def _build_network(self) -> Model:
        """Build Dueling Q-network with separate value and advantage streams."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        # Shared layers
        for units in self.hidden_layers[:-1]:
            x = layers.Dense(units, activation='relu')(x)
        
        # Value stream
        value = layers.Dense(self.hidden_layers[-1], activation='relu')(x)
        value = layers.Dense(1, activation='linear')(value)
        
        # Advantage stream
        advantage = layers.Dense(self.hidden_layers[-1], activation='relu')(x)
        advantage = layers.Dense(self.action_dim, activation='linear')(advantage)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Use Lambda layer to wrap TensorFlow operations
        advantage_mean = layers.Lambda(
            lambda a: tf.reduce_mean(a, axis=1, keepdims=True),
            output_shape=(1,)
        )(advantage)
        advantage_centered = layers.Subtract()([advantage, advantage_mean])
        q_values = layers.Add()([value, advantage_centered])
        
        model = Model(inputs=inputs, outputs=q_values)
        return model


class PrioritizedDDQNAgent(DQNAgent):
    """Prioritized Double DQN Agent with Prioritized Experience Replay."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize Prioritized Double DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary with hyperparameters
        """
        # Initialize base DQN components
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update = config.get('target_update', 10)
        self.hidden_layers = config.get('hidden_layers', [128, 128])
        
        # Prioritized replay parameters
        self.alpha = config.get('alpha', 0.6)  # Priority exponent
        self.beta_start = config.get('beta_start', 0.4)  # Importance sampling exponent
        self.beta_increment = config.get('beta_increment', 0.001)  # Increase beta over time
        self.beta = self.beta_start
        
        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=config.get('memory_size', 10000),
            alpha=self.alpha
        )
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Training metrics
        self.episode_count = 0
        self.current_reward = 0
        self.current_epsilon = self.epsilon
        self.training_steps = 0
    
    def train_step(self):
        """Perform one training step with Prioritized Double DQN update."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from prioritized replay buffer
        sample_result = self.memory.sample(self.batch_size, self.beta)
        if sample_result is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = sample_result
        
        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Double DQN: use policy network to select actions, target network to evaluate
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1, output_type=tf.int32)
        next_q_values = self.target_network(next_states, training=False)
        next_q_values = tf.gather_nd(next_q_values, 
                                     tf.stack([tf.range(self.batch_size, dtype=tf.int32), next_actions], axis=1))
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Train network with importance sampling weights
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values = tf.reduce_sum(q_values * action_masks, axis=1)
            
            # Compute TD errors for priority update
            td_errors = target_q_values - q_values
            
            # Weighted loss using importance sampling
            loss = tf.reduce_mean(weights * tf.square(td_errors))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update priorities in buffer (use absolute TD errors + small constant)
        priorities = np.abs(td_errors.numpy()) + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        return loss.numpy()


class QLearningAgent:
    """Q-Learning Agent (Value-Based, Tabular)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize Q-Learning agent with tabular Q-values."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        
        # State discretization parameters
        self.n_bins = config.get('n_bins', 10)
        self.state_bounds = config.get('state_bounds', [(-1, 1)] * state_dim)
        
        # Create bins for each dimension
        self.bins = [np.linspace(bounds[0], bounds[1], self.n_bins + 1) 
                     for bounds in self.state_bounds]
        
        # Q-table (dictionary for continuous states)
        self.q_table = {}
    
    def _discretize_state(self, state):
        """Discretize continuous state into bins for tabular storage."""
        discretized = []
        for i, value in enumerate(state):
            # Clip state values to bounds
            clipped = np.clip(value, self.state_bounds[i][0], self.state_bounds[i][1])
            # Find which bin it belongs to
            bin_idx = np.digitize(clipped, self.bins[i]) - 1
            # Ensure bin_idx is within valid range
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        state_key = self._discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        return np.argmax(self.q_table[state_key])
    
    def select_action(self, state, training=True):
        """Alias for get_action for consistency."""
        return self.get_action(state, training)
    
    def train_step(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule."""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        target_q = reward + (1 - done) * self.gamma * max_next_q
        
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        return abs(target_q - current_q)  # Return TD error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Not needed for Q-learning."""
        pass


class SARSAAgent:
    """SARSA Agent (Value-Based, Tabular, On-Policy)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize SARSA agent with tabular Q-values."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.2)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.998)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.max_timesteps = config.get('max_timesteps', 500)
        self.update_frequency = config.get('update_frequency', 1)
        self.max_state_action_pairs = config.get('state_action_pairs', 50000)
        
        # State discretization parameters
        self.n_bins = config.get('n_bins', 15)
        self.state_bounds = config.get('state_bounds', [(-1, 1)] * state_dim)
        
        # Create bins for each dimension
        self.bins = [np.linspace(bounds[0], bounds[1], self.n_bins + 1) 
                     for bounds in self.state_bounds]
        
        # Q-table
        self.q_table = {}
        self.training_steps = 0  # Track training steps
        
        # Store next action for SARSA update
        self.next_action = None
    
    def _discretize_state(self, state):
        """Discretize continuous state into bins for tabular storage."""
        discretized = []
        for i, value in enumerate(state):
            # Clip state values to bounds
            clipped = np.clip(value, self.state_bounds[i][0], self.state_bounds[i][1])
            # Find which bin it belongs to
            bin_idx = np.digitize(clipped, self.bins[i]) - 1
            # Ensure bin_idx is within valid range
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        state_key = self._discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            if state_key not in self.q_table:
                # Check if we've reached max state-action pairs limit
                if len(self.q_table) < self.max_state_action_pairs:
                    self.q_table[state_key] = np.zeros(self.action_dim)
                else:
                    # If table is full, use random action for unseen states
                    action = np.random.randint(self.action_dim)
                    return action
            action = np.argmax(self.q_table[state_key])
        
        return action
    
    def select_action(self, state, training=True):
        """Alias for get_action for consistency."""
        return self.get_action(state, training)
    
    def train_step(self, state, action, reward, next_state, done):
        """Update Q-table using SARSA update rule."""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Check state-action pairs limit before adding new states
        if state_key not in self.q_table:
            if len(self.q_table) < self.max_state_action_pairs:
                self.q_table[state_key] = np.zeros(self.action_dim)
            else:
                return 0.0  # Skip update if table is full
        
        if next_state_key not in self.q_table:
            if len(self.q_table) < self.max_state_action_pairs:
                self.q_table[next_state_key] = np.zeros(self.action_dim)
            else:
                return 0.0  # Skip update if table is full
        
        # For SARSA, we need the next action from next_state (on-policy)
        # Get next action for next state
        next_action = self.get_action(next_state, training=True)
        
        # SARSA update: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        current_q = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action]
        target_q = reward + (1 - done) * self.gamma * next_q
        
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        self.training_steps += 1
        
        # Store next action for next iteration (SARSA needs continuity)
        self.next_action = next_action
        
        return abs(target_q - current_q)  # Return TD error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Not needed for SARSA."""
        pass


class REINFORCEAgent:
    """REINFORCE Agent (Gradient-Based Policy Gradient)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize REINFORCE agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.hidden_layers = config.get('hidden_layers', [128, 128])
        
        # Policy network
        self.policy_network = self._build_policy_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Training metrics
        self.episode_count = 0
        self.current_reward = 0
        self.current_epsilon = 0.0  # Not used in policy gradient
    
    def _build_policy_network(self) -> Model:
        """Build policy network."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action by sampling from policy distribution."""
        action_probs = self.policy_network(np.expand_dims(state, 0), training=False)[0]
        action = np.random.choice(self.action_dim, p=action_probs.numpy())
        
        if training:
            self.episode_states.append(state)
            self.episode_actions.append(action)
        
        return action
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.episode_rewards.append(reward)
    
    def train_step(self):
        """Train policy network at end of episode."""
        if len(self.episode_rewards) == 0:
            return None
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Train network
        with tf.GradientTape() as tape:
            loss = 0
            for state, action, G in zip(self.episode_states, self.episode_actions, returns):
                action_probs = self.policy_network(np.expand_dims(state, 0), training=True)
                action_prob = action_probs[0, action]
                loss += -tf.math.log(action_prob + 1e-8) * G
            
            loss = loss / len(self.episode_states)
        
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
        # Clear episode memory
        loss_value = loss.numpy()
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        
        return loss_value
    
    def decay_epsilon(self):
        """Not used in policy gradient methods."""
        pass


class ActorCriticAgent:
    """Actor-Critic Agent (Gradient-Based)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize Actor-Critic agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate_actor = config.get('learning_rate_actor', 0.001)
        self.learning_rate_critic = config.get('learning_rate_critic', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.hidden_layers = config.get('hidden_layers', [128, 128])
        
        # Networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate_actor)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate_critic)
        
        # Training metrics
        self.episode_count = 0
        self.current_reward = 0
        self.current_epsilon = 0.0
    
    def _build_actor(self) -> Model:
        """Build actor (policy) network."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_critic(self) -> Model:
        """Build critic (value) network."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action by sampling from policy."""
        action_probs = self.actor(np.expand_dims(state, 0), training=False)[0]
        action = np.random.choice(self.action_dim, p=action_probs.numpy())
        return action
    
    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step (online learning)."""
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, 0), dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor(np.expand_dims(next_state, 0), dtype=tf.float32)
        reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        # Calculate next value (not part of gradient for critic)
        next_value = 0.0 if done else self.critic(next_state_tensor, training=False)[0, 0]
        target = reward_tensor + self.gamma * next_value
        
        # Update critic
        with tf.GradientTape() as tape:
            value_pred = self.critic(state_tensor, training=True)[0, 0]
            critic_loss = tf.square(target - value_pred)
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Calculate TD error for actor (using the value before update)
        value_current = self.critic(state_tensor, training=False)[0, 0]
        td_error = target - value_current
        
        # Update actor
        with tf.GradientTape() as tape:
            action_probs = self.actor(state_tensor, training=True)
            action_prob = action_probs[0, action]
            actor_loss = -tf.math.log(action_prob + 1e-8) * td_error
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return float(critic_loss.numpy())
    
    def decay_epsilon(self):
        """Not used in policy gradient methods."""
        pass


class PPOAgent:
    """Proximal Policy Optimization Agent (Gradient-Based)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize PPO agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.hidden_layers = config.get('hidden_layers', [64, 64])
        
        # Networks
        self.actor_critic = self._build_actor_critic()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Training metrics
        self.episode_count = 0
        self.current_reward = 0
        self.current_epsilon = 0.0
    
    def _build_actor_critic(self) -> Model:
        """Build combined actor-critic network."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        # Actor output
        actor_output = layers.Dense(self.action_dim, activation='softmax')(x)
        
        # Critic output
        critic_output = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=[actor_output, critic_output])
        return model
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action and store information for training."""
        state_input = np.expand_dims(state, 0)
        action_probs, value = self.actor_critic(state_input, training=False)
        action_probs = action_probs[0].numpy()
        action = np.random.choice(self.action_dim, p=action_probs)
        
        if training:
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value[0, 0].numpy())
            self.log_probs.append(np.log(action_probs[action] + 1e-8))
        
        return action
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train_step(self):
        """Train PPO agent at end of episode."""
        if len(self.rewards) == 0:
            return None
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        G = 0
        
        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + self.gamma * G * (1 - self.dones[i])
            returns.insert(0, G)
            advantages.insert(0, G - self.values[i])
        
        returns = np.array(returns)
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        
        # PPO training epochs
        total_loss = 0
        for _ in range(self.n_epochs):
            with tf.GradientTape() as tape:
                action_probs, values = self.actor_critic(states, training=True)
                
                # Actor loss (clipped)
                action_masks = tf.one_hot(actions, self.action_dim)
                new_log_probs = tf.math.log(tf.reduce_sum(action_probs * action_masks, axis=1) + 1e-8)
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                
                # Critic loss
                critic_loss = tf.reduce_mean(tf.square(returns - values[:, 0]))
                
                # Entropy bonus
                entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            gradients = tape.gradient(loss, self.actor_critic.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))
            total_loss += loss.numpy()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        return total_loss / self.n_epochs
    
    def decay_epsilon(self):
        """Not used in policy gradient methods."""
        pass


class A2CAgent:
    """Advantage Actor-Critic (A2C) Agent (Gradient-Based)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize A2C agent (synchronous version of A3C)."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.n_steps = config.get('n_steps', 5)
        
        # Build actor-critic network
        self.actor_critic = self._build_actor_critic()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Storage for n-step returns
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def _build_actor_critic(self) -> Model:
        """Build combined actor-critic network."""
        inputs = layers.Input(shape=(self.state_dim,))
        
        # Shared layers
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        
        # Actor head (policy)
        actor = layers.Dense(64, activation='relu')(x)
        actor_output = layers.Dense(self.action_dim, activation='softmax')(actor)
        
        # Critic head (value)
        critic = layers.Dense(64, activation='relu')(x)
        critic_output = layers.Dense(1, activation='linear')(critic)
        
        model = Model(inputs=inputs, outputs=[actor_output, critic_output])
        return model
    
    def get_action(self, state, training=True):
        """Select action using stochastic policy."""
        state = np.expand_dims(state, axis=0)
        action_probs, value = self.actor_critic(state, training=False)
        
        if training:
            action = np.random.choice(self.action_dim, p=action_probs.numpy()[0])
        else:
            action = np.argmax(action_probs.numpy()[0])
        
        return action
    
    def select_action(self, state, training=True):
        """Alias for get_action for consistency."""
        return self.get_action(state, training)
    
    def train_step(self, state, action, reward, next_state, done):
        """Perform online training step with n-step returns."""
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
        # Get value estimate
        state_tensor = tf.expand_dims(state, axis=0)
        _, value = self.actor_critic(state_tensor, training=False)
        self.values.append(value.numpy()[0, 0])
        self.dones.append(done)
        
        # Train every n_steps or at episode end
        if len(self.states) >= self.n_steps or done:
            loss = self._update_networks(next_state, done)
            return loss
        
        return None
    
    def _update_networks(self, next_state, done):
        """Update actor-critic networks using n-step returns."""
        # Compute returns
        returns = []
        if done:
            next_value = 0.0
        else:
            next_state_tensor = tf.expand_dims(next_state, axis=0)
            _, next_value_tensor = self.actor_critic(next_state_tensor, training=False)
            next_value = next_value_tensor.numpy()[0, 0]
        
        # Compute n-step returns
        for reward, value, d in zip(reversed(self.rewards), 
                                    reversed(self.values), 
                                    reversed(self.dones)):
            if d:
                next_value = 0.0
            next_value = reward + self.gamma * next_value
            returns.insert(0, next_value)
        
        returns = np.array(returns)
        states = np.array(self.states)
        actions = np.array(self.actions)
        values = np.array(self.values)
        
        # Compute advantages
        advantages = returns - values
        
        # Update networks
        with tf.GradientTape() as tape:
            action_probs, value_preds = self.actor_critic(states, training=True)
            
            # Actor loss (policy gradient with advantage)
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.math.log(tf.reduce_sum(action_probs * action_masks, axis=1) + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * advantages)
            
            # Critic loss (value function)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(value_preds)))
            
            # Entropy bonus for exploration
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        gradients = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))
        
        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
        return total_loss.numpy()
    
    def decay_epsilon(self):
        """Not used in policy gradient methods."""
        pass
    
    def update_target_network(self):
        """Not needed for A2C."""
        pass


class DDPGAgent:
    """Deep Deterministic Policy Gradient (DDPG) Agent (Gradient-Based, Continuous Actions)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize DDPG agent for continuous action spaces."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.actor_lr = config.get('actor_lr', 0.0001)
        self.critic_lr = config.get('critic_lr', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.noise_std = config.get('noise_std', 0.1)
        
        # Build networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Copy weights to target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=config.get('buffer_size', 100000))
    
    def _build_actor(self) -> Model:
        """Build actor network (policy)."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_critic(self) -> Model:
        """Build critic network (Q-function)."""
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        
        # State pathway
        state_out = layers.Dense(256, activation='relu')(state_input)
        
        # Concatenate state and action
        concat = layers.Concatenate()([state_out, action_input])
        x = layers.Dense(256, activation='relu')(concat)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        return model
    
    def get_action(self, state, training=True):
        """Select continuous action with exploration noise."""
        state = np.expand_dims(state, axis=0)
        action = self.actor(state, training=False).numpy()[0]
        
        if training:
            # Add Gaussian noise for exploration
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def select_action(self, state, training=True):
        """Alias for get_action for consistency."""
        return self.get_action(state, training)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states, training=False)
            target_q = self.target_critic([next_states, target_actions], training=False)
            target_q = rewards + (1 - dones) * self.gamma * tf.squeeze(target_q)
            
            current_q = self.critic([states, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(target_q - tf.squeeze(current_q)))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred], training=False))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return critic_loss.numpy()
    
    def update_target_network(self):
        """Soft update of target networks."""
        # Update target actor
        for target_param, param in zip(self.target_actor.weights, self.actor.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        # Update target critic
        for target_param, param in zip(self.target_critic.weights, self.critic.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
    
    def decay_epsilon(self):
        """Gradually reduce exploration noise."""
        self.noise_std = max(0.01, self.noise_std * 0.995)


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent (Gradient-Based, Continuous)."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize TD3 agent - improved DDPG with twin critics and delayed policy updates."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.actor_lr = config.get('actor_lr', 0.0001)
        self.critic_lr = config.get('critic_lr', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.noise_std = config.get('noise_std', 0.1)
        self.policy_delay = config.get('policy_delay', 2)
        self.target_noise = config.get('target_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        
        # Build networks (twin critics)
        self.actor = self._build_actor()
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic_1 = self._build_critic()
        self.target_critic_2 = self._build_critic()
        
        # Copy weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # Optimizers (separate optimizers for each critic)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate=self.critic_lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=config.get('buffer_size', 100000))
        
        # Training counter for delayed policy updates
        self.train_step_counter = 0
    
    def _build_actor(self) -> Model:
        """Build actor network."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_critic(self) -> Model:
        """Build critic network."""
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        
        concat = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(concat)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        return model
    
    def get_action(self, state, training=True):
        """Select continuous action with exploration noise."""
        state = np.expand_dims(state, axis=0)
        action = self.actor(state, training=False).numpy()[0]
        
        if training:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def select_action(self, state, training=True):
        """Alias for get_action for consistency."""
        return self.get_action(state, training)
    
    def train_step(self):
        """Perform one training step with twin critics and delayed policy updates."""
        if len(self.memory) < self.batch_size:
            return None
        
        self.train_step_counter += 1
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Update critics
        with tf.GradientTape(persistent=True) as tape:
            # Target policy smoothing
            target_actions = self.target_actor(next_states, training=False)
            noise = tf.clip_by_value(
                tf.random.normal(shape=target_actions.shape, stddev=self.target_noise),
                -self.noise_clip, self.noise_clip
            )
            target_actions = tf.clip_by_value(target_actions + noise, -1, 1)
            
            # Twin Q-values, take minimum
            target_q1 = self.target_critic_1([next_states, target_actions], training=False)
            target_q2 = self.target_critic_2([next_states, target_actions], training=False)
            target_q = tf.minimum(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * tf.squeeze(target_q)
            
            # Critic losses
            current_q1 = self.critic_1([states, actions], training=True)
            current_q2 = self.critic_2([states, actions], training=True)
            critic_loss_1 = tf.reduce_mean(tf.square(target_q - tf.squeeze(current_q1)))
            critic_loss_2 = tf.reduce_mean(tf.square(target_q - tf.squeeze(current_q2)))
        
        # Update both critics with separate optimizers
        critic_1_grads = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        
        del tape
        
        # Delayed policy update
        if self.train_step_counter % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actions_pred = self.actor(states, training=True)
                actor_loss = -tf.reduce_mean(self.critic_1([states, actions_pred], training=False))
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            # Update target networks
            self.update_target_network()
        
        return (critic_loss_1.numpy() + critic_loss_2.numpy()) / 2
    
    def update_target_network(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.target_actor.weights, self.actor.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        for target_param, param in zip(self.target_critic_1.weights, self.critic_1.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        for target_param, param in zip(self.target_critic_2.weights, self.critic_2.weights):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
    
    def decay_epsilon(self):
        """Gradually reduce exploration noise."""
        self.noise_std = max(0.01, self.noise_std * 0.995)


# Factory function to create agents
def create_agent(method_type: str, method_name: str, state_dim: int, 
                 action_dim: int, config: Dict[str, Any]):
    """
    Factory function to create RL agents.
    Handles method names with instance suffixes (e.g., "DQN_2", "PPO_3").
    
    Args:
        method_type: "value_based" or "gradient_based"
        method_name: Name of the method (may include _N suffix for duplicates)
        state_dim: State space dimension
        action_dim: Action space dimension
        config: Configuration dictionary
        
    Returns:
        Agent instance
    """
    agents = {
        "DQN": DQNAgent,
        "Double_DQN": DoubleDQNAgent,
        "Dueling_DQN": DuelingDQNAgent,
        "Prioritized_DDQN": PrioritizedDDQNAgent,
        "Q_Learning": QLearningAgent,
        "SARSA": SARSAAgent,
        "REINFORCE": REINFORCEAgent,
        "Actor_Critic": ActorCriticAgent,
        "A2C": A2CAgent,
        "PPO": PPOAgent,
        "DDPG": DDPGAgent,
        "TD3": TD3Agent
    }
    
    # Extract base method name (remove _N suffix if present)
    # E.g., "DQN_2" -> "DQN", "REINFORCE_3" -> "REINFORCE"
    base_method_name = method_name
    if '_' in method_name and method_name.split('_')[-1].isdigit():
        base_method_name = '_'.join(method_name.split('_')[:-1])
    
    if base_method_name not in agents:
        raise ValueError(f"Unknown method: {base_method_name} (from {method_name})")
    
    return agents[base_method_name](state_dim, action_dim, config)
