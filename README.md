# RL Training Workbench

A comprehensive reinforcement learning training system with GUI configurator and live visualization. This workbench supports multiple RL algorithms (both value-based and gradient-based) for training agents on Gymnasium environments.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Supported RL Methods](#supported-rl-methods)
- [Configuration](#configuration)
- [Results](#results)
- [Requirements](#requirements)

## Features

- **GUI Configurator**: Easy-to-use interface for selecting environments, methods, and configuring hyperparameters
- **Multiple RL Algorithms**: Support for 12+ RL methods including DQN variants, Q-Learning, SARSA, PPO, A2C, DDPG, and more
- **Live Visualization**: Real-time environment rendering and training metrics plotting
- **Testset Management**: Save/load experiment configurations as JSON testsets
- **Comparative Training**: Train and compare multiple agents simultaneously
- **Modern UI**: Dark-themed interface with intuitive controls
- **Results Tracking**: Automatic saving of training results with timestamps

## Project Structure

```
workbench_final/
├── WB_main.py                    # Main entry point
├── README.md                     # This file
├── src/
│   ├── WB_config.py             # Configuration classes
│   ├── WB_configurator_gui.py   # GUI configurator
│   ├── WB_logic.py              # RL algorithm implementations
│   └── WB_runner.py             # Training runner with visualization
├── testset/                     # Saved experiment configurations
│   └── Acrobot-v1_testset_*.json
└── results/                     # Training results
    └── Acrobot-v1_*/
        └── training_results.json
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install numpy tensorflow gymnasium matplotlib pillow
```

### Optional: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Launch the Configurator**:
   ```bash
   python WB_main.py
   ```

2. **Configure Your Experiment**:
   - Select a Gymnasium environment (e.g., CartPole-v1, Acrobot-v1)
   - Choose one or more RL methods
   - Adjust hyperparameters (or use defaults)
   - Set number of training episodes
   - Save configuration as testset (optional)

3. **Run Training**:
   - Click "RUN" to launch training with live visualization
   - Monitor real-time metrics and environment rendering
   - Training results saved automatically to `results/` directory

### Command Line Usage

```bash
# Run main configurator
python WB_main.py

# Or directly import and use modules
python -c "from src.WB_config import Config_app; config = Config_app(); print(config)"
```

### Example Workflow

```python
# Create configuration
from src.WB_config import Config_app, Config_agent, Config_env

config = Config_app()
config.environment.name = "CartPole-v1"
config.num_episodes = 500

# Add DQN agent
agent = Config_agent(method_type="value_based", method_name="DQN")
agent.parameters = agent.get_default_parameters()
config.add_agent(agent)

# Save testset
config.save_testset()
```

## Supported RL Methods

### Value-Based Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| **DQN** | Deep Q-Network | learning_rate, gamma, epsilon_decay |
| **Double_DQN** | Double DQN (reduces overestimation) | learning_rate, gamma, target_update |
| **Dueling_DQN** | Dueling architecture | hidden_layers, memory_size |
| **Prioritized_DDQN** | Prioritized experience replay | alpha, beta_start, beta_increment |
| **Q_Learning** | Tabular Q-Learning | learning_rate, n_bins, state_bounds |
| **SARSA** | On-policy TD control | learning_rate, epsilon_decay, n_bins |

### Gradient-Based Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| **REINFORCE** | Monte Carlo policy gradient | learning_rate, gamma |
| **Actor_Critic** | Actor-Critic architecture | learning_rate_actor, learning_rate_critic |
| **A2C** | Advantage Actor-Critic | value_coef, entropy_coef |
| **PPO** | Proximal Policy Optimization | clip_epsilon, n_epochs |
| **DDPG** | Deep Deterministic Policy Gradient | tau, batch_size |
| **TD3** | Twin Delayed DDPG | policy_delay, noise_clip |

## Configuration

### Configuration File Structure

Testset configurations are saved as JSON files in the `testset/` directory:

```json
{
    "environment": {
        "name": "Acrobot-v1",
        "render_mode": null,
        "max_episode_steps": 500,
        "seed": null
    },
    "agents": [
        {
            "method_type": "value_based",
            "method_name": "Dueling_DQN",
            "parameters": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 32,
                "memory_size": 100000,
                "target_update": 10,
                "hidden_layers": [128, 128]
            }
        }
    ],
    "num_episodes": 1000,
    "project_name": "RL_Workbench",
    "results_dir": "./results"
}
```

### Key Configuration Classes

- **Config_env**: Environment settings (name, render mode, max steps)
- **Config_agent**: Agent configuration (method type, name, parameters)
- **Config_gui**: GUI settings (window size, animation, theme)
- **Config_app**: Main application configuration combining all components

## Results

Training results are automatically saved to `results/<environment>_<timestamp>/training_results.json`:

```json
{
    "environment": "Acrobot-v1",
    "num_episodes": 1000,
    "timestamp": "2025-11-15T10:30:00",
    "agents": [
        {
            "method_name": "Dueling_DQN",
            "episode_rewards": [...],
            "moving_average": [...],
            "final_avg_reward": -85.23
        }
    ]
}
```

### Viewing Results

- Results are plotted live during training
- Saved JSON files can be analyzed post-training
- Moving average (100-episode window) tracks learning progress

## Requirements

```
numpy>=1.21.0
tensorflow>=2.10.0
gymnasium>=0.28.0
matplotlib>=3.5.0
pillow>=9.0.0
```

## Architecture

### Module Overview

1. **WB_main.py**: Entry point that launches the configurator GUI
2. **WB_config.py**: Configuration management with dataclasses
3. **WB_configurator_gui.py**: Tkinter-based GUI for experiment setup
4. **WB_logic.py**: RL algorithm implementations (agents, replay buffers, neural networks)
5. **WB_runner.py**: Training execution with live visualization

### Key Components

- **ReplayBuffer**: Experience replay for off-policy methods
- **PrioritizedReplayBuffer**: Prioritized experience replay with importance sampling
- **Agent Classes**: Implementations of each RL method
- **TrainingMetrics**: Real-time metrics tracking and plotting
- **RunnerGUI**: Live visualization of training progress

## Syntax Validation

All Python files have been checked and contain **no syntax errors**. The codebase follows Python best practices:

- Type hints for function parameters and returns
- Docstrings for classes and methods
- Dataclasses for configuration management
- Modular architecture with clear separation of concerns

## Contributing

When contributing, please:
1. Follow existing code style and documentation patterns
2. Add type hints to new functions
3. Include docstrings for public methods
4. Test with at least one environment before committing

## License

This project is part of a Reinforcement Learning research project.

## Author

**Katja** - Reinforcement Learning Workbench for Gymnasium Environments

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym)
- Uses [TensorFlow](https://www.tensorflow.org/) for deep learning
- Inspired by classic RL papers and implementations
