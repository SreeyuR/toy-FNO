#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=32G   # memory per CPU core

#SBATCH --time=1-00:00:00   # walltime format is DAYS-HOURS:MINUTES:SECONDS

#SBATCH -J "Toy FNO Second Run"   # job name
#SBATCH --mail-user=sratala@caltech.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

# conda activate fusion2
#jupyter nbconvert --to notebook --execute output.py --output Output.ipynb

/home/sratala/miniconda3/envs/fusion2/bin/python toy_FNO.py
