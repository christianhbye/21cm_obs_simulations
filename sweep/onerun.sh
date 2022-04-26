#!/bin/bash
#SBATCH --time=6:30:00
#SBATCH --nodes=3
#SBATCH --mail-user=<chb@berkeley.edu>
#SBATCH --mail-type=ALL

source $HOME/conv_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/$SCRATCH/global21cm

python run_many_parallel_conv.py 0

