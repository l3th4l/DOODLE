#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate artist
python optuna_env.py --study-name "$1" --trials "$2"  
