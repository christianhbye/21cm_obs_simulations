#!/bin/bash
#SBATCH --time=00:12:00
#SBATCH --array=7-60:1
#SBATCH --mail-user=<christian.bye@mail.mcgill.ca>
#SBATCH --mail-type=ALL

source $HOME/conv_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/$SCRATCH/global21cm

python run_many_parallel_conv.py $SLURM_ARRAY_TASK_ID

