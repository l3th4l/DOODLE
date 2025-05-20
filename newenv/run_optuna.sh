#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate artist
python optuna_env.py --study_name "$1" --n_trials "$2"  
