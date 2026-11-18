#!/bin/bash
#SBATCH -Jed_sample_par
#SBATCH --account=gts-jscott319
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=10     # 10 CPUs on ONE node: sample_parallel.py forks 10 worker processes (reads $SLURM_CPUS_PER_TASK)
#SBATCH --mem-per-cpu=4G
#SBATCH -t10:00:00                                     # estimate ~5 h for 10^6 boxes (see run log); the run is resumable: if it is killed, submit again
#SBATCH -qinferno
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-jscott319-0/jyang872/decomp_srun/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jyang872@gatech.edu

# usage:  sbatch sample_parallel.sbatch            (scenario 1)
#         sbatch sample_parallel.sbatch 2          (scenario 2)
module load anaconda3
conda activate pyomo-binder
which python ipopt                                     # both must resolve (ipopt must be on PATH)
cd /storage/project/r-jscott319-0/jyang872/decomp
python sample_parallel.py ${1:-1}
