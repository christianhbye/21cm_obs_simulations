#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH --array=1-3:1
#SBATCH --nodes=1
#SBATCH --mail-user=<christian.bye@mail.mcgill.ca>
#SBATCH --mail-type=ALL

source $HOME/conv_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/$SCRATCH/global21cm

python run_many_parallel_conv.py $SLURM_ARRAY_TASK_ID

