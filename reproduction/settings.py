# /home/scur0989/vec2text/reproduction/settings.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
OUTPUTS_DIR = PROJECT_ROOT / "reproduction" / "outputs"
VEC2TEXT_DIR = PROJECT_ROOT 

print("Project Root:", PROJECT_ROOT)
print("Outputs Directory:", OUTPUTS_DIR)
print("Vec2Text Directory:", VEC2TEXT_DIR)