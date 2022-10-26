#!/bin/bash
#SBATCH --job-name=OCgri
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=70
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.weiel@kit.edu
#SBATCH --account=haicore-project-kit
#SBATCH --partition=cpuonly

module purge
module restore propulate
source ~/.virtualenvs/PROPULATE/bin/activate

RESDIR=./job_${SLURM_JOB_ID}/
mkdir ${RESDIR}
cd ${RESDIR}
python ../create_db.py griewank ${SLURM_JOB_ID}
srun python ../optimize.py griewank ${SLURM_JOB_ID}
