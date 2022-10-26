#!/bin/bash

# hooray for stack overflow...
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for DeepCam on either HoreKa or Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-f, --framework           which framework to use [optuna, propulate]"
      echo "-N, --nodes               number of nodes to compute on"
      echo "-t, --time                compute time limit"
      echo "-w, --strongweak          is this a strong or weak run [strong, weak]"
      echo "-b, --benchmark           which benchmark function to run"
      exit 0
      ;;
    -f|--framework) shift; export FRAMEWORK=$1; shift; ;;
    -N|--nodes) shift; export SLURM_NNODES=$1; shift; ;;
    -t|--time) shift; export TIMELIMIT=$1; shift; ;;
    -w|--strongweak) shift; export STRONGWEAK=$1; shift; ;;
    -b|--benchmark) shift; export BENCHMARK=$1; shift; ;;
    *) break; ;;
  esac
done

if [ -z "${TIMELIMIT}" ]; then TIMELIMIT="00:10:00"; fi

export BENCHMARK="${BENCHMARK}"
export PYTORCH_KERNEL_CACHE_PATH="${TMP}"

ase_dir="/hkfs/work/workspace/scratch/qv2382-propulate/exps/"
export OUTPUT_ROOT="${base_dir}slurm/"
export OUTPUT_DIR="${OUTPUT_ROOT}"
export FRAMEWORK="${FRAMEWORK}"
echo "${OUTPUT_ROOT}"


echo "Running the $BENCHMARK benchmark (${STRONGWEAK} scaling) with a time limit of time limit of $TIMELIMIT"

#echo "strong/weak: ${STRONGWEAK} \tstage only? ${STAGE_ONLY}"

#echo "${STRONGWEAK} scaling run"

SBATCH_PARAMS=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "70"
  --time               "${TIMELIMIT}"
#  --gres               "gpu:4"
  --job-name           "prop-${FRAMEWORK}-${BENCHMARK}"
  --partition	         "cpuonly"
  --output             "${OUTPUT_DIR}slurm-out-${BENCHMARK}-%j.out"
  -A                   "haicore-project-scc"
)

# no strong/weak here -> only

#if [ "${STRONGWEAK}" == "strong" ];
#  then
#    ngpus="$(( SLURM_NNODES * 4 ))";
#    export TRAINING_INSTANCE_SIZE="${ngpus}"
#    export NINSTANCES=1

#elif [ "${STRONGWEAK}" == "weak" ];
#  then
#    ngpus="${TRAINING_INSTANCE_SIZE}";
#    export NINSTANCES="$(( SLURM_NNODES * 4 / TRAINING_INSTANCE_SIZE ))"
#fi

#sbatch "${SBATCH_PARAMS[@]}" "start_${BENCHMARK}.sh"

#if [ "$BENCHMARK" == "sphere" ]
#  then
#	  sbatch "${SBATCH_PARAMS[@]}" "start__training.sh"
#else
#  echo "must specify system that we are running on!"
#  exit 128
#fi

