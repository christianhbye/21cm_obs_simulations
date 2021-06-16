#!/bin/bash
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --mail-user=<christian.bye@mail.mcgill.ca>
#SBATCH --mail-type=ALL

source $HOME/conv_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/$SCRATCH/global21cm

python run_many_parallel_conv.py 1
python run_many_parallel_conv.py 2
python run_many_parallel_conv.py 3

