#!/bin/bash

#SBATCH --time-min=04:00:00
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH --nodes=32
#SBATCH --constraint="knl"
#SBATCH --output=sample-%j.out
#SBATCH --mem-per-cpu=MaxMemPerNode

python model_AA.py
