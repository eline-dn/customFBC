#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem 42gb
#SBATCH --time 06:00:00
#SBATCH --output=approx_%A.log

#SBATCH --mail-type=ALL

# Initialise environment and modules
#source /work/lpdi/users/mpacesa/Pipelines/miniforge3/bin/activate /work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/FastBC
source /work/lpdi/users/mpacesa/Pipelines/miniforge3/bin/activate /work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma
module load gcc/13.2
module load cuda/12.4.1
module load cudnn/8.9.7.29-12

settings_path="$1"
filters_path="$2"
advanced_path="$3"

python run_approx.py "${settings_path}" "${filters_path}" "${advanced_path}"
