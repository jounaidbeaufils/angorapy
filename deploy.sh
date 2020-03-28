#!/bin/bash

#SBATCH --job-name=ihom_experiment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=admin@tonioweidler.de
#SBATCH --time=00:05:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# this is an adaption of the ray boilerplate https://ray.readthedocs.io/en/latest/deploying-on-slurm.html

worker_num=2 # Must be one less that the total number of nodes

# load modules
module load daint-gpu
module load cray-python
module load cray-nvidia-compute
module load cudatoolkit/10.0.130_3.22-7.0.1.0_5.2__gdfb4ce5
module av graphviz

# load virtual environment
source venv/bin/activate

# get names of allocated nodes and create array
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# choose head node
node1=${nodes_array[0]}

# make head adress
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

# export ip head so pyhton script can read it
export ip_head

# start head node
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password &
sleep 20

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 20
done

python -u test_slurm_deploy.py $redis_password 36 # Pass the total number of allocated CPUs