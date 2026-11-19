#!/bin/bash
#SBATCH -Jed_points
#SBATCH --array=1-3                                    # array task id = scenario (1, 2, 3): three independent jobs, 10 CPUs each; use e.g. --array=1 for scenario 1 only
#SBATCH --account=gts-jscott319
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=10     # sample_points.py forks one worker per CPU (reads $SLURM_CPUS_PER_TASK)
#SBATCH --mem-per-cpu=4G
#SBATCH -t24:00:00                                     # the run is RESUMABLE (finished chunks are skipped): if it is killed or not finished, submit the same command again
#SBATCH -qinferno
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-jscott319-0/jyang872/decomp_srun/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jyang872@gatech.edu

# usage:  sbatch sample_points.sbatch                  (all three scenarios, N_PTS from sample_points.py = 100 per dimension)
#         N_PTS=30 sbatch sample_points.sbatch         (30 points per dimension instead; the environment variable is passed to the job)
#         sbatch --array=1 sample_points.sbatch        (scenario 1 only)
module load anaconda3
conda activate pyomo-binder
which python ipopt                                     # both must resolve (ipopt must be on PATH)
cd /storage/project/r-jscott319-0/jyang872/decomp
python sample_points.py ${SLURM_ARRAY_TASK_ID} ${N_PTS:-100}
