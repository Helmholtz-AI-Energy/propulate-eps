#!/bin/bash
#SBATCH --job-name=OTros
#SBATCH --time=5:00:00
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
python ../create_db.py rosenbrock ${SLURM_JOB_ID}
srun python ../optimize.py rosenbrock ${SLURM_JOB_ID}
