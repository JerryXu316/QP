#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --partition=sugon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=process_data_%A_%a.out
#SBATCH --error=process_data_%A_%a.err
#SBATCH --array=0

source ~/.bashrc
conda activate gurobi

python solve.py initial_generated_data.csv output_part_${SLURM_ARRAY_TASK_ID}.csv