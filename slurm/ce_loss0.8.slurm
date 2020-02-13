#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Sat Mar 02 2019 00:19:42 GMT+1100 (Australian Eastern Daylight Time)
# Multithreaded (SMP) job: must run on one node and the cloud partition
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1

# The name of the job:
#SBATCH --job-name="SCELoss"

# The project ID which this job should run under:
#SBATCH --account="punim0784"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# The amount of memory in megabytes per process in the job:
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time 4:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Run the job from this directory:
cd /data/cephfs/punim0784/SCELoss

# The modules to load:
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA10
nvidia-smi

# CIFAR10
python3 -u train.py  --loss         CE                \
                     --dataset_type cifar10           \
                     --l2_reg       1e-2              \
                     --seed         123               \
                     --version      CE0.8_CIFAR10     \
                     --nr           0.8

# CIFAR100
python3 -u train.py  --lr           0.01              \
                     --loss         CE                \
                     --dataset_type cifar100          \
                     --l2_reg       1e-2              \
                     --seed         123               \
                     --version      CE0.8_CIFAR100    \
                     --nr           0.8
