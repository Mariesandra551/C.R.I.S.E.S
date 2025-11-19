# src/pipeline.py
import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    script = Path(__file__).resolve().parent / script_name
    subprocess.run([sys.executable, str(script)], check=True)

def main():
    run_script("model.py")
    run_script("fill_deficit.py")
    run_script("train_model.py")
    run_script("visualize_predictions.py")  # visualization stays here


if __name__ == "__main__":
    main()
