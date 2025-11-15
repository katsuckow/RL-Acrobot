"""
Config Module - Configuration Classes for RL Training System
Defines configuration classes for application, GUI, environment, and agent settings.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Config_env:
    """Environment configuration for Gymnasium environments."""
    
    name: str = "CartPole-v1"
    render_mode: Optional[str] = None
    max_episode_steps: int = 500
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config_env':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Config_agent:
    """Agent configuration with method-specific parameters."""
    
    method_type: str = "value_based"  # "value_based" or "gradient_based"
    method_name: str = "DQN"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config_agent':
        """Create from dictionary."""
        return cls(**data)
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the selected method."""
        defaults = {
            # Value-based methods
            "DQN": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 20000,
                "target_update": 10,
                "hidden_layers": [128, 128]
            },
            "Double_DQN": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 20000,
                "target_update": 10,
                "hidden_layers": [128, 128]
            },
            "Dueling_DQN": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 20000,
                "target_update": 10,
                "hidden_layers": [128, 128]
            },
            "Prioritized_DDQN": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 20000,
                "target_update": 10,
                "hidden_layers": [128, 128],
                "alpha": 0.6,
                "beta_start": 0.4,
                "beta_increment": 0.001
            },
            "Q_Learning": {
                "learning_rate": 0.1,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "n_bins": 10,                   # Number of bins for state discretization (simpler than SARSA)
                "state_bounds": [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-12.6, 12.6), (-12.6, 12.6)]  # Acrobot state bounds
            },
            "SARSA": {
                "learning_rate": 0.2,           # Learning Rate (α) - Increased for faster learning
                "gamma": 0.99,                  # Discount Factor (γ)
                "epsilon": 1.0,                 # Exploration Rate (initial ε)
                "epsilon_decay": 0.998,         # Exploration Decay Rate - Slower decay for more exploration
                "epsilon_min": 0.01,            # Minimum Exploration Rate
                "max_timesteps": 500,           # Max Timesteps per Episode
                "update_frequency": 1,          # Update Frequency (steps between updates)
                "state_action_pairs": 50000,    # Max State-Action Pairs - Increased for larger Q-table
                "n_bins": 15,                   # Number of bins for state discretization - Finer discretization
                "state_bounds": [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-12.6, 12.6), (-12.6, 12.6)]  # Acrobot state bounds
            },
            # Gradient-based methods
            "REINFORCE": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "hidden_layers": [128, 128]
            },
            "Actor_Critic": {
                "learning_rate_actor": 0.001,
                "learning_rate_critic": 0.001,
                "gamma": 0.99,
                "hidden_layers": [128, 128]
            },
            "A2C": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "hidden_layers": [128, 128]
            },
            "PPO": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "clip_epsilon": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "batch_size": 64,
                "n_epochs": 10,
                "hidden_layers": [64, 64]
            },
            "DDPG": {
                "learning_rate_actor": 0.001,
                "learning_rate_critic": 0.001,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 64,
                "memory_size": 100000,
                "hidden_layers": [400, 300]
            },
            "TD3": {
                "learning_rate_actor": 0.001,
                "learning_rate_critic": 0.001,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_delay": 2,
                "noise_clip": 0.5,
                "batch_size": 64,
                "memory_size": 100000,
                "hidden_layers": [400, 300]
            }
        }
        
        return defaults.get(self.method_name, {})


@dataclass
class Config_gui:
    """GUI configuration settings."""
    
    window_width: int = 1200
    window_height: int = 800
    theme: str = "default"
    animation_enabled: bool = True
    plot_update_frequency: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config_gui':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Config_app:
    """Main application configuration combining all config components."""
    
    environment: Config_env = field(default_factory=Config_env)
    agents: List[Config_agent] = field(default_factory=list)
    gui: Config_gui = field(default_factory=Config_gui)
    
    # Training settings
    num_episodes: int = 1000
    project_name: str = "RL_Workbench"
    experiment_name: str = ""
    testset_dir: str = "./testset"
    results_dir: str = "./results"
    
    def __post_init__(self):
        """Initialize experiment name if not provided."""
        if not self.experiment_name:
            self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_agent(self, agent: Config_agent):
        """Add an agent configuration."""
        self.agents.append(agent)
    
    def remove_agent(self, index: int):
        """Remove an agent configuration by index."""
        if 0 <= index < len(self.agents):
            self.agents.pop(index)
    
    def clear_agents(self):
        """Clear all agent configurations."""
        self.agents.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment.to_dict(),
            'agents': [agent.to_dict() for agent in self.agents],
            'gui': self.gui.to_dict(),
            'num_episodes': self.num_episodes,
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'testset_dir': self.testset_dir,
            'results_dir': self.results_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config_app':
        """Create configuration from dictionary."""
        env_config = Config_env.from_dict(data.get('environment', {}))
        agents_config = [Config_agent.from_dict(agent) for agent in data.get('agents', [])]
        gui_config = Config_gui.from_dict(data.get('gui', {}))
        
        return cls(
            environment=env_config,
            agents=agents_config,
            gui=gui_config,
            num_episodes=data.get('num_episodes', 1000),
            project_name=data.get('project_name', 'RL_Workbench'),
            experiment_name=data.get('experiment_name', ''),
            testset_dir=data.get('testset_dir', './testset'),
            results_dir=data.get('results_dir', './results')
        )
    
    def save_testset(self, filepath: Optional[str] = None) -> str:
        """
        Save configuration as testset JSON file.
        
        Args:
            filepath: Optional custom filepath. If None, generates filename automatically.
            
        Returns:
            Path to saved file.
        """
        if filepath is None:
            # Generate filename: environment_name_testset_XXX.json
            testset_files = [f for f in os.listdir(self.testset_dir) 
                           if f.startswith(self.environment.name) and f.endswith('.json')]
            testset_number = len(testset_files) + 1
            filename = f"{self.environment.name}_testset_{testset_number:03d}.json"
            filepath = os.path.join(self.testset_dir, filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        print(f"Testset saved to {filepath}")
        return filepath
    
    @classmethod
    def load_testset(cls, filepath: str) -> 'Config_app':
        """
        Load configuration from testset JSON file.
        
        Args:
            filepath: Path to testset JSON file.
            
        Returns:
            Config_app instance.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls.from_dict(data)
        print(f"Testset loaded from {filepath}")
        return config
    
    def get_testset_list(self) -> List[str]:
        """
        Get list of available testset files.
        
        Returns:
            List of testset filenames.
        """
        if not os.path.exists(self.testset_dir):
            return []
        
        return sorted([f for f in os.listdir(self.testset_dir) if f.endswith('.json')])
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if valid, False otherwise.
        """
        try:
            assert len(self.agents) > 0, "At least one agent must be configured"
            assert self.num_episodes > 0, "Number of episodes must be positive"
            assert self.environment.name, "Environment name must be specified"
            
            for agent in self.agents:
                assert agent.method_name, "Agent method name must be specified"
                assert agent.method_type in ["value_based", "gradient_based"], \
                    "Method type must be 'value_based' or 'gradient_based'"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation error: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = Config_app()
    config.environment.name = "CartPole-v1"
    config.num_episodes = 500
    
    # Add agents
    agent1 = Config_agent(method_type="value_based", method_name="DQN")
    agent1.parameters = agent1.get_default_parameters()
    config.add_agent(agent1)
    
    agent2 = Config_agent(method_type="gradient_based", method_name="PPO")
    agent2.parameters = agent2.get_default_parameters()
    config.add_agent(agent2)
    
    # Save and load
    if config.validate():
        filepath = config.save_testset()
        loaded_config = Config_app.load_testset(filepath)
        print("\nLoaded configuration:")
        print(json.dumps(loaded_config.to_dict(), indent=2))
