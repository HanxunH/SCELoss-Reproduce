
declare -a target=(
                   "ce_loss"
                   "sce_loss"
                  )

#
declare -a nr_arr=(
                   "0.0"
                   "0.2"
                   "0.4"
                   "0.6"
                   "0.8"
                  )

#
for i in "${target[@]}"
do
  for j in "${nr_arr[@]}"
  do
    job_name=${i}${j}
    echo $job_name
    sbatch --partition deeplearn --nodes 1 --gres=gpu:v100:1 --time 3:00:00 --qos gpgpudeeplearn --job-name $job_name --mem=16G ${job_name}.slurm
  done
done
