#!/bin/bash -l
#SBATCH --job-name=simple   # Job name
#SBATCH --account=project_462000472
#SBATCH --output=output_okvqa_val_beit_9pnp_know_11shots_paper_%j.txt
#SBATCH --error=errors_okvqa_val_beit_9pnp_know_11shots_paper_%j.txt
#SBATCH --partition=small-g       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=48:00:00         # Run time (hh:mm:ss)


module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000472/pyry/python_base

python get_info.py