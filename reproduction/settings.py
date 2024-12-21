from pathlib import Path

# This points to the directory containing settings.py.
# which is vec2text/reproduction
PROJECT_ROOT = Path(__file__).resolve().parent  

# Define other directories relative to the root
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
JOB_SCRIPTS_DIR = PROJECT_ROOT / "job_scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = PROJECT_ROOT / "plots"