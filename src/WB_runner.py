"""
WB_runner.py - RL Training Runner with Live Animation and Plotting
Executes training with real-time visualization of environment and performance metrics.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import gymnasium as gym
import threading
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
from collections import deque
from PIL import Image, ImageTk

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from WB_config import Config_app
from WB_logic import create_agent


class TrainingMetrics:
    """Store and manage training metrics for plotting."""
    
    def __init__(self, method_name: str, color: str):
        """Initialize metrics storage."""
        self.method_name = method_name
        self.color = color
        self.episodes = []
        self.rewards = []
        self.moving_avg = []
        self.window_size = 100
    
    def add_reward(self, episode: int, reward: float):
        """Add reward for an episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        
        # Calculate moving average
        if len(self.rewards) >= self.window_size:
            avg = np.mean(self.rewards[-self.window_size:])
        else:
            avg = np.mean(self.rewards)
        self.moving_avg.append(avg)
        
        # Debug output
        if episode % 10 == 0:  # Only log every 10 episodes to reduce spam
            print(f"[METRICS] {self.method_name}: Episode {episode}, Reward {reward:.2f}, Avg {avg:.2f}")


class RunnerGUI:
    """Main runner GUI with live animation and plotting."""
    
    def __init__(self, root: tk.Tk, testset_filepath: str):
        """
        Initialize runner GUI.
        
        Args:
            root: Tkinter root window
            testset_filepath: Path to testset configuration file
        """
        self.root = root
        self.root.title("RL Training Runner")
        self.root.minsize(1200, 800)
        self.root.state('zoomed')  # Start maximized on Windows
        
        # Modern color palette - Black, Grey, Silver
        self.ui_colors = {
            'bg_dark': '#1a1a1a',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3d3d3d',
            'fg_primary': '#e0e0e0',
            'fg_secondary': '#a0a0a0',
            'accent': '#4a90e2',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336'
        }
        
        # Apply color scheme
        self.root.configure(bg=self.ui_colors['bg_dark'])
        self._apply_color_scheme()
        
        # Load configuration
        self.config = Config_app.load_testset(testset_filepath)
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.env = None
        self.env_render = None  # Environment for rendering
        self.agents = []
        self.metrics = {}
        
        # Colors for different methods (plot lines)
        self.colors = ['#4a90e2', '#f39c12', '#2ecc71', '#e74c3c', '#9b59b6', '#1abc9c']
        
        # Current episode info
        self.current_episode = 0
        self.current_method = ""
        self.current_reward = 0.0
        self.current_epsilon = 0.0
        self.termination_flag = ""
        
        # Animation state
        self.selected_method = None
        self.current_frame = None
        self.current_photo_image = None  # Store reference to prevent garbage collection
        self.agent_configs = []
        self.animation_enabled = True  # Toggle for animation on/off
        
        self._create_widgets()
        self._initialize_plot()
    
    def _apply_color_scheme(self):
        """Apply modern black/grey/silver color scheme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('TFrame', background=self.ui_colors['bg_dark'])
        style.configure('TLabelframe', 
                       background=self.ui_colors['bg_dark'],
                       bordercolor=self.ui_colors['bg_light'])
        style.configure('TLabelframe.Label',
                       background=self.ui_colors['bg_dark'],
                       foreground=self.ui_colors['fg_primary'],
                       font=('Arial', 10, 'bold'))
        style.configure('TLabel',
                       background=self.ui_colors['bg_dark'],
                       foreground=self.ui_colors['fg_primary'])
        style.configure('TButton',
                       background=self.ui_colors['bg_light'],
                       foreground=self.ui_colors['fg_primary'],
                       font=('Arial', 10, 'bold'))
        style.map('TButton',
                 background=[('active', self.ui_colors['accent'])])
        style.configure('TPanedwindow', background=self.ui_colors['bg_dark'])
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main container with two panels
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Animation and controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Plot
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # === LEFT PANEL ===
        
        # Title with custom styling - HARDCODED TO ACROBOT-V1
        title_label = tk.Label(left_frame, 
                              text="Classic Control Environment: Acrobot", 
                              font=('Arial', 16, 'bold'),
                              bg=self.ui_colors['bg_dark'],
                              fg=self.ui_colors['accent'])
        title_label.pack(pady=10)
        
        # Configuration info
        info_frame = ttk.LabelFrame(left_frame, text="Configuration", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = f"Environment: {self.config.environment.name}\n"
        info_text += f"Episodes: {self.config.num_episodes}\n"
        info_text += f"Methods: {', '.join([a.method_name for a in self.config.agents])}"
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack()
        
        # Training status labels
        status_frame = ttk.LabelFrame(left_frame, text="Training Status", padding="10")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.episode_var = tk.StringVar(value="Episode: 0 / 0")
        ttk.Label(status_frame, textvariable=self.episode_var, 
                 font=('Arial', 11, 'bold')).pack(anchor=tk.W)
        
        self.method_var = tk.StringVar(value="Method: -")
        ttk.Label(status_frame, textvariable=self.method_var).pack(anchor=tk.W)
        
        self.reward_var = tk.StringVar(value="Reward: 0.00")
        ttk.Label(status_frame, textvariable=self.reward_var).pack(anchor=tk.W)
        
        self.epsilon_var = tk.StringVar(value="Epsilon: 0.00")
        ttk.Label(status_frame, textvariable=self.epsilon_var).pack(anchor=tk.W)
        
        self.flag_var = tk.StringVar(value="")
        self.flag_label = ttk.Label(status_frame, textvariable=self.flag_var, 
                                    foreground='red', font=('Arial', 10, 'bold'))
        self.flag_label.pack(anchor=tk.W)
        
        # Training diagnostics (shows training is working)
        diag_frame = ttk.LabelFrame(left_frame, text="Training Diagnostics", padding="10")
        diag_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.buffer_var = tk.StringVar(value="Buffer: - / -")
        ttk.Label(diag_frame, textvariable=self.buffer_var, 
                 font=('Arial', 9)).pack(anchor=tk.W)
        
        self.training_steps_var = tk.StringVar(value="Training Steps: 0")
        ttk.Label(diag_frame, textvariable=self.training_steps_var, 
                 font=('Arial', 9)).pack(anchor=tk.W)
        
        self.exploration_var = tk.StringVar(value="Exploration: -")
        ttk.Label(diag_frame, textvariable=self.exploration_var, 
                 font=('Arial', 9)).pack(anchor=tk.W)
        
        # Animation canvas with dark theme
        animation_frame = ttk.LabelFrame(left_frame, text="Environment Animation", padding="10")
        animation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Dropdown for method selection and animation toggle (above animation)
        dropdown_frame = ttk.Frame(animation_frame)
        dropdown_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(dropdown_frame, text="Select Method:", 
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.method_dropdown = ttk.Combobox(dropdown_frame, state='readonly', width=20)
        self.method_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.method_dropdown.bind('<<ComboboxSelected>>', self._on_method_selected)
        
        # Animation toggle button
        self.toggle_anim_button = ttk.Button(dropdown_frame, text="Animation: ON", 
                                             command=self._toggle_animation, width=15)
        self.toggle_anim_button.pack(side=tk.LEFT, padx=5)
        
        # Parameters label (shows parameters of selected method)
        self.params_var = tk.StringVar(value="Parameters will be shown here")
        params_label = ttk.Label(animation_frame, textvariable=self.params_var, 
                                font=('Arial', 8), wraplength=400)
        params_label.pack(fill=tk.X, pady=(0, 5))
        
        self.animation_canvas = tk.Canvas(animation_frame, 
                                         bg=self.ui_colors['bg_medium'], 
                                         highlightthickness=0,
                                         width=400, height=300)
        self.animation_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Animation info text
        self.anim_info_var = tk.StringVar(value="Animation will appear here during training")
        ttk.Label(animation_frame, textvariable=self.anim_info_var, 
                 font=('Arial', 9, 'italic')).pack()
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Training", 
                                       command=self._start_training)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training", 
                                      command=self._stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # === RIGHT PANEL ===
        
        # Plot title
        plot_title = tk.Label(right_frame, 
                             text="Training Performance", 
                             font=('Arial', 16, 'bold'),
                             bg=self.ui_colors['bg_dark'],
                             fg=self.ui_colors['accent'])
        plot_title.pack(pady=10)
        
        # Matplotlib figure with dark theme
        self.fig = Figure(figsize=(8, 7), dpi=100, facecolor=self.ui_colors['bg_dark'])
        self.ax = self.fig.add_subplot(111, facecolor=self.ui_colors['bg_medium'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg=self.ui_colors['bg_dark'], highlightthickness=0)
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _initialize_plot(self):
        """Initialize the matplotlib plot with dark theme."""
        self.ax.set_xlabel('Episode', fontsize=12, color=self.ui_colors['fg_primary'])
        self.ax.set_ylabel('Reward', fontsize=12, color=self.ui_colors['fg_primary'])
        self.ax.set_title('Episode Rewards', fontsize=14, fontweight='bold', 
                         color=self.ui_colors['accent'])
        self.ax.grid(True, alpha=0.2, color=self.ui_colors['fg_secondary'])
        self.ax.tick_params(colors=self.ui_colors['fg_primary'])
        self.ax.spines['bottom'].set_color(self.ui_colors['fg_secondary'])
        self.ax.spines['top'].set_color(self.ui_colors['fg_secondary'])
        self.ax.spines['left'].set_color(self.ui_colors['fg_secondary'])
        self.ax.spines['right'].set_color(self.ui_colors['fg_secondary'])
        # Legend will be added when data is plotted
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _on_method_selected(self, event=None):
        """Handle method selection from dropdown."""
        selected = self.method_dropdown.get()
        if selected:
            self.selected_method = selected  # Update selected method
            # Find the agent config for the selected method
            for agent_config in self.config.agents:
                if agent_config.method_name == selected:
                    # Display parameters
                    params_str = ", ".join([f"{k}={v}" for k, v in agent_config.parameters.items()])
                    self.params_var.set(f"Parameters: {params_str}")
                    break
    
    def _toggle_animation(self):
        """Toggle animation on/off during training."""
        self.animation_enabled = not self.animation_enabled
        
        # Update button text and status
        if self.animation_enabled:
            self.toggle_anim_button.config(text="Animation: ON")
            self.anim_info_var.set("Animation enabled")
        else:
            self.toggle_anim_button.config(text="Animation: OFF")
            self.anim_info_var.set("Animation disabled")
            # Clear canvas when disabled
            self.animation_canvas.delete("all")
            self.animation_canvas.create_text(
                200, 150,
                text="Animation Disabled\n(Toggle to enable)",
                fill=self.ui_colors['fg_secondary'],
                font=('Arial', 12, 'italic')
            )
    
    def _populate_dropdown(self, method_names: List[str]):
        """Populate dropdown with available methods."""
        try:
            self.method_dropdown['values'] = method_names
            if method_names:
                self.method_dropdown.current(0)
                self.selected_method = method_names[0]  # Initialize selected method
                self._on_method_selected()
        except Exception as e:
            print(f"Error populating dropdown: {e}")
    
    def _update_plot(self):
        """Update the plot with current metrics."""
        try:
            print(f"[DEBUG] _update_plot called")
            print(f"[DEBUG] Metrics keys: {list(self.metrics.keys())}")
            for method_name, metrics in self.metrics.items():
                print(f"[DEBUG] {method_name}: episodes={len(metrics.episodes)}, rewards={len(metrics.rewards)}, avg={len(metrics.moving_avg)}")
            
            self.ax.clear()
            self.ax.set_facecolor(self.ui_colors['bg_medium'])
            
            # Plot each method
            lines_plotted = 0
            for method_name, metrics in self.metrics.items():
                if len(metrics.episodes) > 0:
                    print(f"[DEBUG] Plotting {method_name} with {len(metrics.episodes)} episodes")
                    # Plot raw rewards with transparency
                    self.ax.plot(metrics.episodes, metrics.rewards, 
                               color=metrics.color, alpha=0.3, linewidth=1,
                               label=f'{method_name} (raw)')
                    lines_plotted += 1
                    
                    # Plot moving average with bold line
                    if len(metrics.moving_avg) > 0:
                        self.ax.plot(metrics.episodes, metrics.moving_avg, 
                                   color=metrics.color, linewidth=2.5,
                                   label=f'{method_name} (avg)')
                        lines_plotted += 1
            
            print(f"[DEBUG] Total lines plotted: {lines_plotted}")
            
            self.ax.set_xlabel('Episode', fontsize=12, color=self.ui_colors['fg_primary'])
            self.ax.set_ylabel('Reward', fontsize=12, color=self.ui_colors['fg_primary'])
            self.ax.set_title('Episode Rewards', fontsize=14, fontweight='bold',
                             color=self.ui_colors['accent'])
            self.ax.grid(True, alpha=0.2, color=self.ui_colors['fg_secondary'])
            self.ax.tick_params(colors=self.ui_colors['fg_primary'])
            self.ax.spines['bottom'].set_color(self.ui_colors['fg_secondary'])
            self.ax.spines['top'].set_color(self.ui_colors['fg_secondary'])
            self.ax.spines['left'].set_color(self.ui_colors['fg_secondary'])
            self.ax.spines['right'].set_color(self.ui_colors['fg_secondary'])
            # Only add legend if there are labeled lines
            if self.ax.get_legend_handles_labels()[0]:
                legend = self.ax.legend(loc='upper left', fontsize=9, 
                                       facecolor=self.ui_colors['bg_light'],
                                       edgecolor=self.ui_colors['fg_secondary'],
                                       labelcolor=self.ui_colors['fg_primary'])
                legend.get_frame().set_alpha(0.9)
            self.fig.tight_layout()
            self.canvas.draw()
            print(f"[DEBUG] Canvas drawn, flushing events")  # Debug
            self.canvas.flush_events()  # Force immediate redraw
            # Force the canvas widget to update immediately
            self.canvas.get_tk_widget().update_idletasks()
            print(f"[DEBUG] Plot update complete")  # Debug
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_training(self):
        """Start training in a separate thread."""
        self.is_training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Initializing training...")
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._run_training, daemon=True)
        self.training_thread.start()
    
    def _stop_training(self):
        """Stop training."""
        self.is_training = False
        try:
            self.status_var.set("Stopping training...")
        except:
            pass  # Window may be closing
    
    def _run_training(self):
        """Execute training loop (runs in separate thread)."""
        try:
            # HARDCODED: Only Acrobot-v1 environment allowed
            env_name = "Acrobot-v1"
            max_steps = self.config.environment.max_episode_steps
            print(f"[RUNNER] Creating environment with max_episode_steps={max_steps}")
            self.env = gym.make(env_name, render_mode=None, max_episode_steps=max_steps)
            
            # Create separate environment for rendering
            try:
                self.env_render = gym.make(env_name, render_mode='rgb_array', max_episode_steps=max_steps)
            except Exception as e:
                print(f"Warning: Could not create render environment: {e}")
                self.env_render = None
            
            state_dim = self.env.observation_space.shape[0]
            
            # Check if action space is discrete or continuous
            if hasattr(self.env.action_space, 'n'):
                # Discrete action space
                action_dim = self.env.action_space.n
                is_discrete = True
                self.root.after(0, lambda: self.status_var.set(
                    f"Environment: {env_name} - Discrete action space (dim={action_dim})"
                ))
            else:
                # Continuous action space (Box)
                action_dim = self.env.action_space.shape[0]
                is_discrete = False
                self.root.after(0, lambda ad=action_dim: self.status_var.set(
                    f"Environment: {env_name} - Continuous action space (dim={ad})"
                ))
            
            # Initialize agents and metrics (track agent-config mapping)
            self.agents = []  # Reset agents list for new training run
            self.agent_configs = []  # Store configs for agents that are created
            self.metrics = {}  # Reset metrics for new training run
            
            for i, agent_config in enumerate(self.config.agents):
                color = self.colors[i % len(self.colors)]
                
                # Check if method is compatible with action space
                is_discrete = hasattr(self.env.action_space, 'n')
                method_name = agent_config.method_name
                
                # Skip DDPG and TD3 for discrete spaces, skip others for continuous
                if not is_discrete and method_name not in ['DDPG', 'TD3']:
                    self.root.after(0, lambda m=method_name: self.status_var.set(
                        f"Skipping {m}: requires discrete action space"
                    ))
                    continue
                elif is_discrete and method_name in ['DDPG', 'TD3']:
                    self.root.after(0, lambda m=method_name: self.status_var.set(
                        f"Skipping {m}: requires continuous action space"
                    ))
                    continue
                
                try:
                    agent = create_agent(
                        agent_config.method_type,
                        agent_config.method_name,
                        state_dim,
                        action_dim,
                        agent_config.parameters
                    )
                    # Add training counter for diagnostics
                    agent.training_steps = 0
                    
                    self.agents.append(agent)
                    self.agent_configs.append(agent_config)  # Track corresponding config
                    self.metrics[agent_config.method_name] = TrainingMetrics(
                        agent_config.method_name, color
                    )
                    print(f"[INIT] Created agent: {agent_config.method_name}, color={color}, index={i}")
                    # Debug: Show which agent was created
                    self.root.after(0, lambda m=method_name: self.status_var.set(
                        f"✓ Created agent: {m}"
                    ))
                except Exception as e:
                    self.root.after(0, lambda m=method_name, err=str(e): self.status_var.set(
                        f"✗ Failed to create {m}: {err}"
                    ))
                    print(f"Error creating {method_name} agent: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Check if any agents were created
            if len(self.agents) == 0:
                self.root.after(0, lambda: self.status_var.set(
                    "Error: No compatible agents for this environment"
                ))
                self.root.after(0, self._training_complete)
                return
            
            # Populate dropdown with created agents
            method_names = [config.method_name for config in self.agent_configs]
            self.root.after(0, lambda names=method_names: self._populate_dropdown(names))
            
            self.root.after(0, lambda: self.status_var.set("Training started - All methods running simultaneously"))
            
            # Train all agents simultaneously episode by episode
            for episode in range(1, self.config.num_episodes + 1):
                if not self.is_training:
                    break
                
                self.current_episode = episode
                
                # Train each agent for one episode
                for agent_idx, agent in enumerate(self.agents):
                    if not self.is_training:
                        break
                    
                    agent_config = self.agent_configs[agent_idx]
                    method_name = agent_config.method_name
                    
                    # Extract base method name (remove _N suffix if present)
                    base_method_name = method_name
                    if '_' in method_name and method_name.split('_')[-1].isdigit():
                        base_method_name = '_'.join(method_name.split('_')[:-1])
                    
                    self.root.after(0, lambda m=method_name, e=episode: 
                                  self.method_var.set(f"Episode {e} - Training: {m}"))
                    
                    episode_reward = 0
                    state, _ = self.env.reset()
                    done = False
                    step_count = 0
                    # Use configured max steps per episode (not hardcoded!)
                    max_steps = self.config.environment.max_episode_steps
                    
                    # Reset render environment if it exists
                    if self.env_render is not None:
                        render_state, _ = self.env_render.reset()
                    
                    if episode == 1:  # Print once at start
                        print(f"[TRAINING LOOP] Using max_steps={max_steps} per episode for {method_name}")
                    print(f"[Episode {episode}] Starting {method_name}...")  # Debug
                    
                    # Episode loop for this agent
                    while not done and step_count < max_steps and self.is_training:
                        # Select action
                        action = agent.select_action(state, training=True)
                        
                        # Ensure action is valid for the action space
                        if hasattr(self.env.action_space, 'n'):
                            # Discrete: ensure integer
                            action = int(action)
                        else:
                            # Continuous: ensure numpy array
                            if not isinstance(action, np.ndarray):
                                action = np.array([action])
                        
                        # Take step in environment
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated
                        episode_reward += reward
                        step_count += 1
                        
                        # Update reward display in real-time (every 5 steps to avoid GUI overhead)
                        if step_count % 5 == 0:
                            self.root.after(0, lambda r=episode_reward, s=step_count: 
                                          self.reward_var.set(f"Reward: {r:.2f} (Step {s})"))
                        
                        # Update render environment (if exists and method is currently selected)
                        if self.env_render is not None and self.selected_method == method_name:
                            self.env_render.step(action)
                            # Render frame periodically (every 3 steps to reduce overhead)
                            if step_count % 3 == 0:
                                self.root.after(0, self._render_frame)
                        
                        # Training based on agent type (use base_method_name for type checks)
                        if base_method_name in ['DDPG', 'TD3']:
                            # DDPG/TD3: Add to buffer and train every step
                            agent.memory.add(state, action, reward, next_state, done)
                            if len(agent.memory) >= agent.batch_size:
                                loss = agent.train_step()
                                agent.training_steps += 1
                                # DDPG needs manual target update, TD3 does it internally
                                if base_method_name == 'DDPG':
                                    agent.update_target_network()
                        
                        elif base_method_name in ['DQN', 'Double_DQN', 'Dueling_DQN', 'Prioritized_DDQN']:
                            # DQN variants: Add to buffer and train every step
                            agent.memory.add(state, action, reward, next_state, done)
                            if len(agent.memory) >= agent.batch_size:
                                loss = agent.train_step()
                                agent.training_steps += 1
                        
                        elif base_method_name in ['REINFORCE', 'PPO']:
                            # Episodic methods: Store rewards, train at end of episode
                            import inspect
                            if 'done' in inspect.signature(agent.store_reward).parameters:
                                agent.store_reward(reward, done)
                            else:
                                agent.store_reward(reward)
                        
                        elif base_method_name in ['Actor_Critic', 'A2C', 'Q_Learning', 'SARSA']:
                            # Step-based methods with parameters
                            loss = agent.train_step(state, action, reward, next_state, done)
                            if loss is not None:
                                agent.training_steps += 1
                        
                        state = next_state
                        
                        # Update animation periodically
                        if step_count % 10 == 0:
                            self._update_animation_info(step_count)
                    
                    # Post-episode processing
                    
                    # Train episodic methods at end of episode
                    if base_method_name in ['REINFORCE', 'PPO']:
                        loss = agent.train_step()
                        agent.training_steps += 1
                    
                    # Update target networks for DQN variants (periodic)
                    if base_method_name in ['DQN', 'Double_DQN', 'Dueling_DQN', 'Prioritized_DDQN']:
                        if hasattr(agent, 'target_update') and episode % agent.target_update == 0:
                            agent.update_target_network()
                    
                    # Decay exploration
                    if hasattr(agent, 'decay_epsilon'):
                        agent.decay_epsilon()
                    
                    # Update metrics
                    self.metrics[method_name].add_reward(episode, episode_reward)
                    
                    print(f"[Episode {episode}] {method_name} finished: reward={episode_reward:.2f}, steps={step_count}")  # Debug
                    
                    # Update GUI with diagnostics
                    epsilon = getattr(agent, 'epsilon', 0.0)
                    noise_std = getattr(agent, 'noise_std', 0.0)
                    exploration = epsilon if epsilon > 0 else noise_std
                    termination = "Done" if done else "Max Steps"
                    
                    # Get agent diagnostics
                    buffer_size = len(agent.memory) if hasattr(agent, 'memory') else 0
                    buffer_capacity = getattr(getattr(agent, 'memory', None), 'capacity', 0)
                    training_steps = getattr(agent, 'training_steps', 0)
                    
                    self.root.after(0, lambda m=method_name, e=episode, r=episode_reward, exp=exploration, t=termination, s=step_count, bs=buffer_size, bc=buffer_capacity, ts=training_steps: 
                                  self._update_gui_status(e, m, r, exp, f"{t} ({s} steps)", bs, bc, ts))
                    
                    # Update plot after each agent completes (for synchronized visualization)
                    print(f"[DEBUG] Scheduling plot update after {method_name} episode {episode}")
                    self.root.after(0, self._update_plot)
                    self.root.update()  # Process plot update immediately
                    time.sleep(0.05)  # Brief pause to allow visualization
                
                # Status update showing progress (episode completed for all methods)
                methods_str = ", ".join([config.method_name for config in self.agent_configs])
                self.root.after(0, lambda e=episode, total=self.config.num_episodes, m=methods_str: 
                              self.status_var.set(f"Episode {e}/{total} completed - All methods"))
                
                # Plot already updated after each agent, so just process any remaining events
                self.root.update_idletasks()
            
            # All training complete
            self.root.after(0, lambda: self.status_var.set("All methods training completed!"))
            
            # Close environments
            self.env.close()
            if self.env_render is not None:
                self.env_render.close()
            
            self.root.after(0, self._training_complete)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, self._training_complete)
    
    def _update_gui_status(self, episode: int, method: str, reward: float, 
                          epsilon: float, termination: str, 
                          buffer_size: int = 0, buffer_capacity: int = 0, 
                          training_steps: int = 0):
        """Update GUI status labels with diagnostics."""
        try:
            self.episode_var.set(f"Episode: {episode} / {self.config.num_episodes}")
            self.method_var.set(f"Method: {method}")
            self.reward_var.set(f"Reward: {reward:.2f}")
            self.epsilon_var.set(f"Epsilon: {epsilon:.4f}")
            
            # Update diagnostics
            if buffer_capacity > 0:
                self.buffer_var.set(f"Buffer: {buffer_size} / {buffer_capacity}")
            else:
                self.buffer_var.set(f"Buffer: N/A (episodic method)")
            
            self.training_steps_var.set(f"Training Steps: {training_steps}")
            
            # Show exploration percentage
            if epsilon > 0:
                self.exploration_var.set(f"Exploration: {epsilon*100:.1f}% random")
            else:
                self.exploration_var.set(f"Exploration: Deterministic")
            
            # Show termination flag briefly
            if termination:
                self.flag_var.set(f"⚠ {termination}")
                self.root.after(500, lambda: self.flag_var.set(""))
        except:
            pass  # Window may be closing
    
    def _update_animation_info(self, step: int):
        """Update animation information."""
        self.anim_info_var.set(f"Step: {step}")
    
    def _render_frame(self):
        """Render current frame from environment and display on canvas."""
        try:
            # Skip rendering if animation is disabled
            if not self.animation_enabled:
                return
            
            if self.env_render is None:
                return
            
            # Get frame from render environment
            frame = self.env_render.render()
            
            if frame is None:
                return
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame)
            
            # Get canvas dimensions
            canvas_width = self.animation_canvas.winfo_width()
            canvas_height = self.animation_canvas.winfo_height()
            
            # Skip if canvas not properly initialized
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Resize image to fit canvas while maintaining aspect ratio
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            canvas_aspect = canvas_width / canvas_height
            
            if aspect_ratio > canvas_aspect:
                # Image is wider than canvas
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                # Image is taller than canvas
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Store reference to prevent garbage collection
            self.current_photo_image = photo
            
            # Clear canvas and display image
            self.animation_canvas.delete("all")
            self.animation_canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=photo, 
                anchor=tk.CENTER
            )
            
        except Exception as e:
            print(f"Error rendering frame: {e}")
    
    def _training_complete(self):
        """Clean up after training completes."""
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Training complete!")
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save training results to file."""
        try:
            # Create results directory
            results_dir = os.path.join(self.config.results_dir, 
                                      f"{self.config.environment.name}_{int(time.time())}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save metrics
            results = {}
            for method_name, metrics in self.metrics.items():
                results[method_name] = {
                    'episodes': metrics.episodes,
                    'rewards': metrics.rewards,
                    'moving_avg': metrics.moving_avg
                }
            
            results_file = os.path.join(results_dir, 'training_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save plot
            plot_file = os.path.join(results_dir, 'training_plot.png')
            self.fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            
            self.status_var.set(f"Results saved to: {results_dir}")
            
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main entry point for runner."""
    if len(sys.argv) > 1:
        testset_filepath = sys.argv[1]
    else:
        # For testing, use a default testset
        testset_filepath = "./testset/CartPole-v1_testset_001.json"
        if not os.path.exists(testset_filepath):
            print("No testset file specified or found.")
            print("Usage: python WB_runner.py <testset_filepath>")
            return
    
    root = tk.Tk()
    app = RunnerGUI(root, testset_filepath)
    root.mainloop()


if __name__ == "__main__":
    main()
