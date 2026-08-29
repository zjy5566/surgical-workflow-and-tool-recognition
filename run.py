"""
GenAI Statement: 
Developed with the assistance of Gemini.
Functionality:
1. Automated pipeline orchestrator.
2. Sequential execution: Task A -> Baselines -> Multi-Task Variants -> Testing -> Visualization.
3. Manages Python pathing for subdirectory script execution.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_path, description):
    """Executes a python script and monitors its status."""
    print(f"\n{'='*20} STARTING: {description} {'='*20}")
    print(f"Executing: {script_path}")
    
    # Ensure the project root is in the python path so sub-scripts can import base modules
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    try:
        # Using subprocess to run scripts as separate processes to avoid memory leakage 
        # and namespace collisions between different training configurations.
        result = subprocess.run([sys.executable, script_path], env=env, check=True)
        print(f"{'='*20} FINISHED: {description} {'='*20}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
        sys.exit(1)

def main():
    start_time = datetime.now()
    print(f"Project Pipeline Started at {start_time}")

    # --- Step 1: Training Phase ---
    # Define training scripts in logical order of dependency
    train_scripts = [
        (os.path.join("train", "train_A.py"), "Task A: Phase Recognition Training"),
        (os.path.join("train", "train_baseline.py"), "Task B: Visual Baseline Training"),
        (os.path.join("train", "Timed_Label_Guided.py"), "Task B: Single-Task Label-Guided Training"),
        (os.path.join("train", "Timed_Multi_Task_Label-Guided.py"), "Task B: Multi-Task Label-Guided Training"),
        (os.path.join("train", "Timed_Multi_Task_Pred-Guided.py"), "Task B: Multi-Task Prediction-Guided Training"),
    ]

    for script, desc in train_scripts:
        if os.path.exists(script):
            run_script(script, desc)
        else:
            print(f"Warning: Script {script} not found. Skipping...")

    # --- Step 2: Testing Phase ---
    # Runs comprehensive evaluation across all saved checkpoints
    if os.path.exists("test.py"):
        run_script("test.py", "Global Model Evaluation and Metric Logging")

    # --- Step 3: Visualization Phase ---
    # Generates timeline plots for qualitative analysis
    if os.path.exists("visualization.py"):
        run_script("visualization.py", "Qualitative Result Visualization")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Full Pipeline Completed successfully!")
    print(f"Started: {start_time}")
    print(f"Ended: {end_time}")
    print(f"Total Duration: {duration}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()