#!/bin/bash
#SBATCH --job-name=ijustworkhere
#SBATCH --account=eecs448w24_class
#SBATCH --partition=standard
#SBATCH --time=1:00:00
#SBATCH --mail-type=END

module purge

pip3 install --user pandas
pip3 install --user nltk

python3 scripts/preprocess_text.py
