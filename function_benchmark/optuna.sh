#!/bin/bash

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

#echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"

printf 'Start1: %s %s\n' "$(date)"

#export expid = $((SLURM_JOBID * 1))
# only need this for optuna...
if [ "$SLURM_PROCID" == "0" ]  # && "$FRAMEWORK" == "optuna" ]
then
  echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"
  mysqld_safe --defaults-file="$SQL_CONFIG" & #--bind-address $SLURM_LAUNCH_NODE_IPADDR &
  sleep 3
  echo "${FNAME}-${SLURM_JOBID}"
  mysql --defaults-file="$SQL_CONFIG" -e "CREATE DATABASE IF NOT EXISTS ${FNAME}_${SLURM_JOBID}"
  optuna create-study --study-name "${FNAME}-${SLURM_JOBID}" --storage "mysql://root@localhost/${FNAME}_${SLURM_JOBID}"
else
  sleep 8
fi

#mysql --defaults-file="$SQL_CONFIG" --host $SLURM_LAUNCH_NODE_IPADDR
#echo "Running ${FNAME} benchmark"
#echo "Completed MySQL on host $HOSTNAME"
#mysql --defaults-file="$SQL_CONFIG" &  #-e "CREATE DATABASE IF NOT EXISTS example"
#optuna create-study --study-name "bigearth" --storage "mysql://root@localhost/${SQL_DATA_DIR}"
#export STUDDY_NUMBER=1
python optimize.py

#printf '\n\n'
#printf 'Start2: %s %s\n' "$(date)"
#printf '\n\n'

#if [ "$SLURM_PROCID" == "0" ]  # && "$FRAMEWORK" == "optuna" ]
#then
#  #echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"
#  #mysqld_safe --defaults-file="$SQL_CONFIG" & #--bind-address $SLURM_LAUNCH_NODE_IPADDR &
#  #sleep 3
#  #echo "${FNAME}-${SLURM_JOBID}"
#  mysql --defaults-file="$SQL_CONFIG" -e "CREATE DATABASE IF NOT EXISTS ${FNAME}_$((SLURM_JOBID*2))"
#  optuna create-study --study-name "${FNAME}-$((SLURM_JOBID*2))" --storage "mysql://root@localhost/${FNAME}_$((SLURM_JOBID*2))"
#else
#  sleep 3
#fi
#export STUDDY_NUMBER=2
#python optimize.py

#printf '\n\n'
#printf 'Start 3: %s %s\n' "$(date)"
#if [ "$SLURM_PROCID" == "0" ]  # && "$FRAMEWORK" == "optuna" ]
#then
#  #echo "$SLURM_SRUN_COMM_HOST $SLURM_LAUNCH_NODE_IPADDR"
#  #mysqld_safe --defaults-file="$SQL_CONFIG" & #--bind-address $SLURM_LAUNCH_NODE_IPADDR &
#  #sleep 3
#  #echo "${FNAME}-${SLURM_JOBID}"
#  mysql --defaults-file="$SQL_CONFIG" -e "CREATE DATABASE IF NOT EXISTS ${FNAME}_$((SLURM_JOBID*3))"
#  optuna create-study --study-name "${FNAME}-$((SLURM_JOBID*3))" --storage "mysql://root@localhost/${FNAME}_$((SLURM_JOBID*3))"
#else
#  sleep 3
#fi

#export STUDDY_NUMBER=3
#python optimize.py

