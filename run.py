import os
import sys
import subprocess

# Set the PYTHONPATH to the root directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Define the path to the Streamlit app
app_path = os.path.join('app', 'simulation_ui.py')

# Run the Streamlit app using subprocess
subprocess.run(['streamlit', 'run', app_path])