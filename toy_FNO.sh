#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G   # memory per CPU core

#SBATCH --time=1-00:00:00   # walltime format is DAYS-HOURS:MINUTES:SECONDS

#SBATCH -J "Toy FNO Second Run"   # job name
#SBATCH --mail-user=sratala@caltech.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

/home/sratala/miniconda3/envs/fusion/bin/python toy_FNO_v2.py
