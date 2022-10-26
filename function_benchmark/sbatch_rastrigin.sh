#!/usr/bin/env bash

# Slurm job configuration
## #SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=0
#SBATCH --time=2:00:00
#SBATCH --partition=cpuonly
#SBATCH --account=haicore-project-scc

#SBATCH --job-name=optuna-rastrigin
#SBATCH --output="/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/rastrigin/optuna-%j.out"
## #SBATCH --job-name=propulate-rastrigin
## #SBATCH --output="/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/rastrigin/propulate-%j.out"

ml purge

export FNAME="rastrigin"

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  --label
#  --cpu-bind="ldoms"
)
export FRAMEWORK="optuna"
export EVALS_PER_WORKER=100

export DATA_DIR="/hkfs/work/workspace/scratch/qv2382-bigearthnet/"
export BASE_DIR="/hkfs/work/workspace/scratch/qv2382-propulate/"

export SQL_DATA_DIR="${BASE_DIR}sqldata/optuna"
export SQL_CONFIG="${BASE_DIR}bigearthnet_kit/my.cnf"
export SQL_SOCKET="${BASE_DIR}mysqld.sock"
touch "$SQL_SOCKET"
#export SQL_DATA_DIR="${BASE_DIR}sqldata/optuna/"
#export SQL_CONFIG="${BASE_DIR}bigearthnet_kit/my.cnf"
export SQL_SOCKET_DIR="${BASE_DIR}bigearthnet_kit/mysql/"

export SEED="${RANDOM}"

CONTAINER_DIR="${BASE_DIR}containers/"
SINGULARITY_FILE="${CONTAINER_DIR}scratch-tf-sql.sif"
echo "${SINGULARITY_FILE}"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#singularity instance start --bind ${SQL_DATA_DIR}:/var/lib/mysql --bind ${SQL_SOCKET_DIR}:/run/mysqld "${SINGULARITY_FILE}" mysql

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${BASE_DIR}","${DATA_DIR}","/scratch","$TMP",${SQL_DATA_DIR}:/var/lib/mysql,${SQL_SOCKET_DIR}:/run/mysqld \
  --bind "${SQL_SOCKET_DIR}/var/log/mysql/":/var/log/mysql \
  ${SINGULARITY_FILE} \
  bash optuna.sh

