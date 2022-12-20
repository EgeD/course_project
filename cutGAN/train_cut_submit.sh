#!/bin/bash
# 
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=cut_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --mem-per-cpu=8G
#SBATCH --time=56:00:00
#SBATCH --output=cut_baseline-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=edincer16@ku.edu.tr

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Load Python 3.6.3
echo "Activating Python 3.6.3..."
module load cuda/11.4
module load cudnn/8.2.2/cuda-11.4
module load anaconda/3.6
source activate cyclegan

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

python train.py


