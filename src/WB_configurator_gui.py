"""
WB_configurator_gui.py - Configurator GUI for RL Training System
Provides interface for selecting environments, methods, and configuring parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
import sys
from typing import Optional, Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from WB_config import Config_app, Config_agent, Config_env


class ParameterPopup:
    """Pop-up window for configuring method parameters."""
    
    def __init__(self, parent, method_name: str, method_type: str, 
                 default_params: Dict[str, Any], callback):
        """
        Initialize parameter popup.
        
        Args:
            parent: Parent window
            method_name: Name of the RL method
            method_type: "value_based" or "gradient_based"
            default_params: Dictionary of default parameter values
            callback: Function to call when parameters are applied
        """
        self.parent = parent
        self.method_name = method_name
        self.method_type = method_type
        self.default_params = default_params.copy()
        self.callback = callback
        
        # Colors
        self.colors = {
            'bg_dark': '#1a1a1a',
            'bg_light': '#3d3d3d',
            'fg_primary': '#e0e0e0',
            'accent': '#4a90e2'
        }
        
        # Create popup window
        self.window = tk.Toplevel(parent)
        self.window.title(f"Parameters - {method_name}")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.configure(bg=self.colors['bg_dark'])
        
        # Parameter variables
        self.param_vars = {}
        
        self._create_widgets()
        
        # Make window adaptive to content
        self.window.update_idletasks()
        self.window.minsize(500, 400)
        self.window.geometry("")  # Let it size to content
    
    def _create_widgets(self):
        """Create popup widgets."""
        # Title with custom styling
        title_label = tk.Label(self.window, 
                              text=f"{self.method_name} Parameters", 
                              font=('Arial', 14, 'bold'),
                              bg=self.colors['bg_dark'],
                              fg=self.colors['accent'])
        title_label.pack(pady=10)
        
        # Parameters frame with scrollbar
        canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        params_frame = ttk.Frame(canvas)
        
        params_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=params_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create parameter entries
        row = 0
        for param_name, param_value in self.default_params.items():
            # Parameter label
            label = ttk.Label(params_frame, text=f"{param_name}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
            
            # Parameter entry
            if isinstance(param_value, bool):
                var = tk.BooleanVar(value=param_value)
                entry = ttk.Checkbutton(params_frame, variable=var)
            elif isinstance(param_value, int):
                var = tk.IntVar(value=param_value)
                entry = ttk.Entry(params_frame, textvariable=var, width=30)
            elif isinstance(param_value, float):
                var = tk.DoubleVar(value=param_value)
                entry = ttk.Entry(params_frame, textvariable=var, width=30)
            elif isinstance(param_value, list):
                var = tk.StringVar(value=str(param_value))
                entry = ttk.Entry(params_frame, textvariable=var, width=30)
            else:
                var = tk.StringVar(value=str(param_value))
                entry = ttk.Entry(params_frame, textvariable=var, width=30)
            
            entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
            self.param_vars[param_name] = (var, type(param_value))
            
            row += 1
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons frame
        button_frame = ttk.Frame(self.window)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Reset to default button
        reset_btn = ttk.Button(button_frame, text="Reset to Default", 
                              command=self._reset_to_default)
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        # Apply button
        apply_btn = ttk.Button(button_frame, text="Add and Apply", 
                              command=self._apply_parameters)
        apply_btn.pack(side=tk.RIGHT, padx=10)
    
    def _reset_to_default(self):
        """Reset all parameters to default values."""
        for param_name, (var, param_type) in self.param_vars.items():
            default_value = self.default_params[param_name]
            if param_type == list:
                var.set(str(default_value))
            else:
                var.set(default_value)
    
    def _apply_parameters(self):
        """Apply parameters and close window."""
        try:
            # Extract parameter values
            params = {}
            for param_name, (var, param_type) in self.param_vars.items():
                if param_type == list:
                    # Parse list from string
                    params[param_name] = eval(var.get())
                else:
                    params[param_name] = var.get()
            
            # Call callback with parameters
            self.callback(self.method_name, self.method_type, params)
            
            # Close window
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying parameters: {str(e)}")


class TestsetViewPopup:
    """Pop-up window for viewing loaded testset information."""
    
    def __init__(self, parent, config: Config_app):
        """
        Initialize testset view popup.
        
        Args:
            parent: Parent window
            config: Configuration to display
        """
        # Colors
        self.colors = {
            'bg_dark': '#1a1a1a',
            'bg_light': '#3d3d3d',
            'fg_primary': '#e0e0e0',
            'accent': '#4a90e2'
        }
        
        self.window = tk.Toplevel(parent)
        self.window.title("Testset Information")
        self.window.transient(parent)
        self.window.configure(bg=self.colors['bg_dark'])
        
        self._create_widgets(config)
        
        # Make window adaptive to content
        self.window.update_idletasks()
        self.window.minsize(600, 400)
        self.window.geometry("")  # Let it size to content
    
    def _create_widgets(self, config: Config_app):
        """Create popup widgets."""
        # Title with custom styling
        title_label = tk.Label(self.window, 
                              text="Testset Configuration", 
                              font=('Arial', 14, 'bold'),
                              bg=self.colors['bg_dark'],
                              fg=self.colors['accent'])
        title_label.pack(pady=10)
        
        # Information text with custom colors
        text_widget = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, 
                                               height=20, width=70,
                                               bg=self.colors['bg_light'],
                                               fg=self.colors['fg_primary'],
                                               insertbackground=self.colors['fg_primary'],
                                               font=('Consolas', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Format configuration information
        info = f"Environment: {config.environment.name}\n"
        info += f"Number of Episodes: {config.num_episodes}\n"
        info += f"Experiment Name: {config.experiment_name}\n\n"
        info += "=" * 60 + "\n\n"
        
        for i, agent in enumerate(config.agents, 1):
            info += f"Method {i}: {agent.method_name} ({agent.method_type})\n"
            info += "Parameters:\n"
            for param_name, param_value in agent.parameters.items():
                info += f"  - {param_name}: {param_value}\n"
            info += "\n" + "-" * 60 + "\n\n"
        
        text_widget.insert(tk.END, info)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ttk.Button(self.window, text="Close", command=self.window.destroy)
        close_btn.pack(pady=10)


class ConfiguratorGUI:
    """Main configurator GUI for RL training system."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize configurator GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("RL Training Configurator")
        self.root.minsize(700, 600)
        
        # Modern color palette - Black, Grey, Silver
        self.colors = {
            'bg_dark': '#1a1a1a',      # Dark black background
            'bg_medium': '#2d2d2d',    # Medium grey
            'bg_light': '#3d3d3d',     # Light grey
            'fg_primary': '#e0e0e0',   # Silver text
            'fg_secondary': '#a0a0a0', # Medium grey text
            'accent': '#4a90e2',       # Blue accent
            'accent_hover': '#5da3f5', # Lighter blue
            'border': '#555555',       # Border grey
            'success': '#4caf50',      # Green
            'warning': '#ff9800',      # Orange
            'error': '#f44336'         # Red
        }
        
        # Apply color scheme
        self._apply_color_scheme()
        
        # Configuration
        self.config = Config_app()
        self.config.testset_dir = os.path.abspath("./testset")
        # Set environment to Acrobot-v1 with optimal max steps
        self.config.environment.name = "Acrobot-v1"
        
        # Available methods (all methods)
        self.all_value_based_methods = ["DQN", "Double_DQN", "Dueling_DQN", "Prioritized_DDQN", "Q_Learning", "SARSA"]
        self.all_gradient_based_methods = ["REINFORCE", "Actor_Critic", "A2C", "PPO", "DDPG", "TD3"]
        
        # Methods that require continuous action spaces
        self.continuous_only_methods = ["DDPG", "TD3"]
        
        # Currently available methods (filtered by environment)
        self.value_based_methods = self.all_value_based_methods.copy()
        self.gradient_based_methods = self.all_gradient_based_methods.copy()
        
        # Gymnasium environments (only Acrobot-v1)
        self.gym_environments = [
            "Acrobot-v1"
        ]
        
        # Optimal max steps per episode for each environment
        # Based on environment characteristics and typical solution lengths
        self.optimal_max_steps = {
            # Classic Control
            "CartPole-v1": 500,           # Typically solves in 100-200 steps
            "Acrobot-v1": 500,            # Can need 200-500 steps to swing up
            "MountainCar-v0": 200,        # Typically solves in 100-150 steps
            "Pendulum-v1": 200,           # Short episodes, continuous control
            
            # Box2D
            "LunarLander-v2": 1000,       # Needs time for careful landing
            "LunarLander-v3": 1000,       # Variant with wind
            "BipedalWalker-v3": 1600,     # Complex locomotion task
            "BipedalWalker-v3 (continuous)": 1600,
            "CarRacing-v2": 1000,         # Racing task
            
            # Atari (if added later)
            "Breakout-v4": 10000,         # Long Atari episodes
            "Pong-v4": 10000,
            "SpaceInvaders-v4": 10000,
            
            # Default fallback
            "default": 500
        }
        
        # Track method instances for duplicate naming
        self.method_counter = {}  # e.g., {"DQN": 2, "PPO": 3} means DQN_2, PPO_3 are next
        
        self._create_widgets()
    
    def _get_optimal_max_steps(self, env_name: str) -> int:
        """Get optimal max steps per episode for a given environment."""
        return self.optimal_max_steps.get(env_name, self.optimal_max_steps["default"])
    
    def _is_continuous_environment(self, env_name: str) -> bool:
        """Check if environment has continuous action space."""
        return "(continuous)" in env_name
    
    def _filter_methods_by_environment(self, env_name: str):
        """Filter available methods based on environment action space."""
        is_continuous = self._is_continuous_environment(env_name)
        
        if is_continuous:
            # Continuous environments: allow all methods
            self.value_based_methods = self.all_value_based_methods.copy()
            self.gradient_based_methods = self.all_gradient_based_methods.copy()
        else:
            # Discrete environments: exclude continuous-only methods
            self.value_based_methods = self.all_value_based_methods.copy()
            self.gradient_based_methods = [m for m in self.all_gradient_based_methods 
                                          if m not in self.continuous_only_methods]
        
        # Update dropdowns
        self.value_dropdown['values'] = self.value_based_methods
        self.gradient_dropdown['values'] = self.gradient_based_methods
        
        # Clear current selections if they're no longer valid
        if self.value_method_var.get() not in self.value_based_methods:
            self.value_method_var.set('')
        if self.gradient_method_var.get() not in self.gradient_based_methods:
            self.gradient_method_var.set('')
    
    def _apply_color_scheme(self):
        """Apply modern black/grey/silver color scheme to the GUI."""
        # Configure root window
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base for customization
        
        # Configure Frame
        style.configure('TFrame', background=self.colors['bg_dark'])
        
        # Configure LabelFrame
        style.configure('TLabelframe', 
                       background=self.colors['bg_dark'],
                       bordercolor=self.colors['border'],
                       relief='solid')
        style.configure('TLabelframe.Label',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['fg_primary'],
                       font=('Arial', 10, 'bold'))
        
        # Configure Label
        style.configure('TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['fg_primary'],
                       font=('Arial', 9))
        
        # Configure Button
        style.configure('TButton',
                       background=self.colors['bg_light'],
                       foreground=self.colors['fg_primary'],
                       bordercolor=self.colors['border'],
                       relief='flat',
                       font=('Arial', 9, 'bold'))
        style.map('TButton',
                 background=[('active', self.colors['accent']),
                           ('pressed', self.colors['accent_hover'])],
                 foreground=[('active', '#ffffff')])
        
        # Configure Combobox
        style.configure('TCombobox',
                       fieldbackground=self.colors['bg_light'],
                       background=self.colors['bg_light'],
                       foreground=self.colors['fg_primary'],
                       bordercolor=self.colors['border'],
                       arrowcolor=self.colors['fg_primary'])
        
        # Configure Entry
        style.configure('TEntry',
                       fieldbackground=self.colors['bg_light'],
                       foreground=self.colors['fg_primary'],
                       bordercolor=self.colors['border'],
                       insertcolor=self.colors['fg_primary'])
        
        # Configure Checkbutton
        style.configure('TCheckbutton',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['fg_primary'],
                       font=('Arial', 9, 'bold'))
        
        # Configure Scrollbar
        style.configure('Vertical.TScrollbar',
                       background=self.colors['bg_light'],
                       troughcolor=self.colors['bg_dark'],
                       bordercolor=self.colors['border'],
                       arrowcolor=self.colors['fg_primary'])
    
    def _create_widgets(self):
        """Create main GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title with custom styling
        title_label = tk.Label(main_frame, 
                              text="RL Training Workbench - Acrobot-v1", 
                              font=('Arial', 18, 'bold'),
                              bg=self.colors['bg_dark'],
                              fg=self.colors['accent'],
                              pady=10)
        title_label.pack(pady=10)
        
        # Environment configuration (fixed to Acrobot-v1)
        env_frame = ttk.LabelFrame(main_frame, text="Environment Configuration", padding="10")
        env_frame.pack(fill=tk.X, pady=10)
        
        # Environment fixed to Acrobot-v1
        self.env_var = tk.StringVar(value="Acrobot-v1")
        
        ttk.Label(env_frame, text="Environment:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(env_frame, text="Acrobot-v1 (Fixed - Discrete Action Space)", 
                 font=('Arial', 10, 'bold'), 
                 foreground=self.colors['accent']).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Number of episodes
        ttk.Label(env_frame, text="Number of Episodes:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.episodes_var = tk.IntVar(value=1000)
        ttk.Entry(env_frame, textvariable=self.episodes_var, width=32).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Max steps per episode
        ttk.Label(env_frame, text="Max Steps per Episode:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        optimal_steps = self._get_optimal_max_steps(self.env_var.get())
        self.max_steps_var = tk.IntVar(value=optimal_steps)
        max_steps_entry = ttk.Entry(env_frame, textvariable=self.max_steps_var, width=32)
        max_steps_entry.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Add tooltip/help text for max steps (shows optimal value)
        ttk.Label(env_frame, text=f"(Optimal for {self.env_var.get()}: {optimal_steps}, adjust lower for faster training)", 
                 font=('Arial', 8, 'italic'), 
                 foreground=self.colors['fg_secondary']).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Method selection frame
        methods_frame = ttk.LabelFrame(main_frame, text="Method Selection", padding="10")
        methods_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create two columns for value-based and gradient-based
        left_frame = ttk.Frame(methods_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(methods_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Value-based methods
        self.value_based_var = tk.BooleanVar(value=False)
        value_check = ttk.Checkbutton(left_frame, text="Value Based", 
                                     variable=self.value_based_var,
                                     command=self._toggle_value_based)
        value_check.pack(anchor=tk.W, pady=5)
        
        self.value_methods_frame = ttk.Frame(left_frame)
        self.value_methods_frame.pack(fill=tk.BOTH, expand=True)
        
        self.value_method_var = tk.StringVar()
        self.value_dropdown = ttk.Combobox(self.value_methods_frame, 
                                          textvariable=self.value_method_var,
                                          values=self.value_based_methods, 
                                          width=25, state='readonly')
        self.value_dropdown.bind('<<ComboboxSelected>>', self._on_value_method_selected)
        
        # Gradient-based methods
        self.gradient_based_var = tk.BooleanVar(value=False)
        gradient_check = ttk.Checkbutton(right_frame, text="Gradient Based", 
                                        variable=self.gradient_based_var,
                                        command=self._toggle_gradient_based)
        gradient_check.pack(anchor=tk.W, pady=5)
        
        self.gradient_methods_frame = ttk.Frame(right_frame)
        self.gradient_methods_frame.pack(fill=tk.BOTH, expand=True)
        
        self.gradient_method_var = tk.StringVar()
        self.gradient_dropdown = ttk.Combobox(self.gradient_methods_frame, 
                                             textvariable=self.gradient_method_var,
                                             values=self.gradient_based_methods, 
                                             width=25, state='readonly')
        self.gradient_dropdown.bind('<<ComboboxSelected>>', self._on_gradient_method_selected)
        
        # Selected methods display
        selected_frame = ttk.LabelFrame(main_frame, text="Selected Methods", padding="10")
        selected_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(selected_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.selected_listbox = tk.Listbox(selected_frame, height=6, 
                                           font=('Courier New', 9),
                                           yscrollcommand=scrollbar.set)
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.selected_listbox.yview)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        # Load previous button
        load_btn = ttk.Button(buttons_frame, text="Load Previous", 
                             command=self._load_previous)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear methods button
        clear_btn = ttk.Button(buttons_frame, text="Clear Methods", 
                              command=self._clear_methods)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Save testset button
        save_btn = ttk.Button(buttons_frame, text="Save Testset", 
                             command=self._save_testset)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Apply and start button
        start_btn = ttk.Button(buttons_frame, text="Apply and Start", 
                              command=self._apply_and_start, 
                              style='Accent.TButton')
        start_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize method filtering for default environment
        self._filter_methods_by_environment(self.env_var.get())
        
        # Initialize config with GUI values
        self.config.environment.max_episode_steps = self.max_steps_var.get()
        self.config.num_episodes = self.episodes_var.get()
    
    def _on_environment_changed(self, event=None):
        """Handle environment selection change."""
        env_name = self.env_var.get()
        self.config.environment.name = env_name
        
        # Update max steps to optimal value for this environment
        optimal_steps = self._get_optimal_max_steps(env_name)
        self.max_steps_var.set(optimal_steps)
        
        # Filter methods based on environment action space
        self._filter_methods_by_environment(env_name)
        
        # Update status
        is_continuous = self._is_continuous_environment(env_name)
        action_type = "continuous" if is_continuous else "discrete"
        self.status_var.set(f"Environment: {env_name} ({action_type} action space) - Max steps: {optimal_steps}")
    
    def _on_env_changed(self, event=None):
        """Handle environment selection change (deprecated, use _on_environment_changed)."""
        self._on_environment_changed(event)
    
    def _toggle_value_based(self):
        """Toggle value-based methods dropdown."""
        if self.value_based_var.get():
            self.value_dropdown.pack(pady=5)
        else:
            self.value_dropdown.pack_forget()
            self.value_method_var.set('')
    
    def _toggle_gradient_based(self):
        """Toggle gradient-based methods dropdown."""
        if self.gradient_based_var.get():
            self.gradient_dropdown.pack(pady=5)
        else:
            self.gradient_dropdown.pack_forget()
            self.gradient_method_var.set('')
    
    def _on_value_method_selected(self, event=None):
        """Handle value-based method selection."""
        method_name = self.value_method_var.get()
        if method_name:
            agent = Config_agent(method_type="value_based", method_name=method_name)
            default_params = agent.get_default_parameters()
            ParameterPopup(self.root, method_name, "value_based", 
                          default_params, self._add_method)
    
    def _on_gradient_method_selected(self, event=None):
        """Handle gradient-based method selection."""
        method_name = self.gradient_method_var.get()
        if method_name:
            agent = Config_agent(method_type="gradient_based", method_name=method_name)
            default_params = agent.get_default_parameters()
            ParameterPopup(self.root, method_name, "gradient_based", 
                          default_params, self._add_method)
    
    def _format_params_for_display(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for display in listbox (show key parameters only)."""
        # Key parameters to display for different methods
        key_params = ['learning_rate', 'gamma', 'epsilon_start', 'batch_size', 
                      'hidden_layers', 'buffer_size', 'n_steps']
        
        display_parts = []
        for key in key_params:
            if key in parameters:
                value = parameters[key]
                # Format value nicely
                if isinstance(value, float):
                    display_parts.append(f"{key}={value:.4f}")
                elif isinstance(value, list):
                    display_parts.append(f"{key}={value}")
                else:
                    display_parts.append(f"{key}={value}")
                
                # Limit to first 3 parameters to avoid overcrowding
                if len(display_parts) >= 3:
                    break
        
        if display_parts:
            return ", ".join(display_parts)
        else:
            # If no key parameters found, show first 3 parameters
            items = list(parameters.items())[:3]
            return ", ".join([f"{k}={v}" for k, v in items])
    
    def _add_method(self, method_name: str, method_type: str, parameters: Dict[str, Any]):
        """Add method to configuration with unique naming for duplicates."""
        # Track method instances and generate unique name
        if method_name not in self.method_counter:
            self.method_counter[method_name] = 1
        else:
            self.method_counter[method_name] += 1
        
        # Generate unique method name (e.g., DQN, DQN_2, DQN_3)
        if self.method_counter[method_name] == 1:
            unique_method_name = method_name
        else:
            unique_method_name = f"{method_name}_{self.method_counter[method_name]}"
        
        # Create agent config with unique name
        agent = Config_agent(method_type=method_type, method_name=unique_method_name, 
                           parameters=parameters)
        self.config.add_agent(agent)
        
        # Format parameters for display (limit to key parameters)
        param_display = self._format_params_for_display(parameters)
        
        # Update listbox with unique method name and key parameters
        self.selected_listbox.insert(tk.END, 
                                     f"{unique_method_name} - {param_display}")
        
        self.status_var.set(f"Added method: {unique_method_name}")
        
        # Reset selections
        self.value_method_var.set('')
        self.gradient_method_var.set('')
    
    def _clear_methods(self):
        """Clear all selected methods."""
        if messagebox.askyesno("Confirm", "Clear all selected methods?"):
            self.config.clear_agents()
            self.selected_listbox.delete(0, tk.END)
            self.method_counter.clear()  # Reset method counter
            self.status_var.set("All methods cleared")
    
    def _save_testset(self):
        """Save current configuration as testset."""
        if len(self.config.agents) == 0:
            messagebox.showwarning("Warning", "No methods selected. Add at least one method.")
            return
        
        try:
            # Update configuration
            self.config.environment.name = self.env_var.get()
            self.config.num_episodes = self.episodes_var.get()
            self.config.environment.max_episode_steps = self.max_steps_var.get()
            
            # Save testset
            filepath = self.config.save_testset()
            messagebox.showinfo("Success", f"Testset saved to:\n{filepath}")
            self.status_var.set(f"Testset saved: {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save testset: {str(e)}")
    
    def _load_previous(self):
        """Load previous testset."""
        # Open file dialog starting from testset directory
        testset_dir = self.config.testset_dir
        if not os.path.exists(testset_dir):
            os.makedirs(testset_dir)
        
        filepath = filedialog.askopenfilename(
            title="Load Testset",
            initialdir=testset_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Load configuration
                self.config = Config_app.load_testset(filepath)
                
                # Update GUI
                self.env_var.set(self.config.environment.name)
                self.episodes_var.set(self.config.num_episodes)
                self.max_steps_var.set(self.config.environment.max_episode_steps)
                
                # Rebuild method counter from loaded agents
                self.method_counter.clear()
                for agent in self.config.agents:
                    # Extract base method name and instance number
                    method_name = agent.method_name
                    if '_' in method_name and method_name.split('_')[-1].isdigit():
                        base_name = '_'.join(method_name.split('_')[:-1])
                        instance_num = int(method_name.split('_')[-1])
                        self.method_counter[base_name] = max(
                            self.method_counter.get(base_name, 0), instance_num
                        )
                    else:
                        self.method_counter[method_name] = max(
                            self.method_counter.get(method_name, 0), 1
                        )
                
                # Update selected methods listbox with parameters
                self.selected_listbox.delete(0, tk.END)
                for agent in self.config.agents:
                    param_display = self._format_params_for_display(agent.parameters)
                    self.selected_listbox.insert(tk.END, 
                                                f"{agent.method_name} - {param_display}")
                
                # Show testset information
                TestsetViewPopup(self.root, self.config)
                
                self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load testset: {str(e)}")
    
    def _apply_and_start(self):
        """Apply configuration and start training."""
        if len(self.config.agents) == 0:
            messagebox.showwarning("Warning", "No methods selected. Add at least one method.")
            return
        
        # Update configuration
        self.config.environment.name = self.env_var.get()
        self.config.num_episodes = self.episodes_var.get()
        self.config.environment.max_episode_steps = self.max_steps_var.get()
        
        print(f"[CONFIGURATOR] Starting training with max_episode_steps={self.config.environment.max_episode_steps}")
        
        # Validate configuration
        if not self.config.validate():
            messagebox.showerror("Error", "Configuration validation failed.")
            return
        
        # Save testset before starting
        try:
            filepath = self.config.save_testset()
            self.status_var.set("Starting training...")
            
            # Launch runner
            self._launch_runner(filepath)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def _launch_runner(self, testset_filepath: str):
        """Launch the runner with the current testset."""
        # Import and launch runner
        try:
            from WB_runner import RunnerGUI
            
            # Create new window for runner
            runner_window = tk.Toplevel(self.root)
            runner_app = RunnerGUI(runner_window, testset_filepath)
            
            self.status_var.set("Runner launched")
            
        except ImportError as e:
            messagebox.showerror("Error", 
                               f"Runner module not found. Please ensure WB_runner.py exists.\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch runner: {str(e)}")


def main():
    """Main entry point for configurator GUI."""
    root = tk.Tk()
    app = ConfiguratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
