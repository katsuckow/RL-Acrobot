"""
WB_main.py - Main Entry Point for RL Training Workbench
Launches the configurator GUI to set up and run RL experiments.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.WB_configurator_gui import main

if __name__ == "__main__":
    print("=" * 60)
    print("RL Training Workbench - Configurator")
    print("=" * 60)
    print("\nStarting configurator GUI...")
    print("Use this interface to:")
    print("  - Select Gymnasium environments")
    print("  - Choose RL methods (value-based or gradient-based)")
    print("  - Configure hyperparameters")
    print("  - Save/load testset configurations")
    print("  - Launch training runs")
    print("\n" + "=" * 60 + "\n")
    
    main()
