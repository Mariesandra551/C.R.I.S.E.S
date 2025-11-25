# src/pipeline.py
import subprocess
import sys
from pathlib import Path



"""
Master execution script for the full financial crisis modeling workflow. This single entry point ensures 
full reproducibility. By running one command, the entire data-cleaning, feature engineering, modeling, and visualization
pipeline is executed end-to-end.

Workflow — runs all scripts in sequence:
1. model.py  
      • Reads all Excel sheets  
      • Restructures raw data  
      • Generates `merged_cleaned_dataset.csv`

2. fill_deficit.py  
      • Interpolates missing values  
      • Applies median country-level filling  
      • Outputs `merged_cleaned_dataset_filled.csv`

3. train_model.py  
      • Creates economic shock features  
      • Fits GMM crisis regime model  
      • Trains early-warning classifier  
      • Generates model outputs and alert files

4. visualize_predictions.py  
      • Creates crisis trend visualizations  
      • Generates contagion matrices and timelines  
      • Supports dashboard & reporting
"""

def run_script(script_name):
    script = Path(__file__).resolve().parent / script_name
    subprocess.run([sys.executable, str(script)], check=True)

def main():
    run_script("model.py")
    run_script("fill_deficit.py")
    run_script("train_model.py")
    run_script("visualize_predictions.py")



if __name__ == "__main__":
    main()
