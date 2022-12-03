#!/bin/bash

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

#echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"

printf 'Start1: %s %s\n' "$(date)"

#export expid = $((SLURM_JOBID * 1))
# only need this for optuna...
if [ "$FRAMEWORK" == "optuna" ]
then
  if [ "$SLURM_PROCID" == "0" ]  # && "$FRAMEWORK" == "optuna" ]
  then
    echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"
    mysqld_safe --defaults-file="$SQL_CONFIG" & #--bind-address $SLURM_LAUNCH_NODE_IPADDR &
    sleep 5
    echo "${FNAME}-${SLURM_JOBID}"
    mysql --defaults-file="$SQL_CONFIG" -e "CREATE DATABASE IF NOT EXISTS ${FNAME}_${SLURM_JOBID}"
    optuna create-study --study-name "${FNAME}-${SLURM_JOBID}" --storage "mysql://root@localhost/${FNAME}_${SLURM_JOBID}"
  else
    sleep 8
  fi
fi

#mysql --defaults-file="$SQL_CONFIG" --host $SLURM_LAUNCH_NODE_IPADDR
#echo "Running ${FNAME} benchmark"
#echo "Completed MySQL on host $HOSTNAME"
#mysql --defaults-file="$SQL_CONFIG" &  #-e "CREATE DATABASE IF NOT EXISTS example"
#optuna create-study --study-name "bigearth" --storage "mysql://root@localhost/${SQL_DATA_DIR}"
#export STUDDY_NUMBER=1
python -u optimize.py

if [ "$FRAMEWORK" == "optuna" ]
then
  if [ "$SLURM_PROCID" == "0" ]  # && "$FRAMEWORK" == "optuna" ]
  then
    service mysql stop
  fi
fi

