#!/bin/bash

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"


# only need this for optuna...
if [ "$SLURM_PROCID" == "0" && "$FRAMEWORK" == "optuna" ]
then
  mysqld_safe --defaults-file="$SQL_CONFIG" & #--bind-address $SLURM_LAUNCH_NODE_IPADDR &
  sleep 3
else
  sleep 8
fi

#mysql --defaults-file="$SQL_CONFIG" --host $SLURM_LAUNCH_NODE_IPADDR
#echo "Running ${FNAME} benchmark"
#echo "Completed MySQL on host $HOSTNAME"
#mysql --defaults-file="$SQL_CONFIG" &  #-e "CREATE DATABASE IF NOT EXISTS example"
#optuna create-study --study-name "bigearth" --storage "mysql://root@localhost/${SQL_DATA_DIR}"
python -u optimize.py
