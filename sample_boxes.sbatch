#!/bin/bash
#SBATCH -Jed_boxes
#SBATCH --array=0-2                       # task 0,1,2 -> scenario 1,2,3
#SBATCH --account=gts-jscott319
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH -t01:00:00                        # 64 boxes x 5 s BARON limit = ~6 min worst case (raise if you raise MaxTime / MAX_BOXES)
#SBATCH -qinferno
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-jscott319-0/jyang872/CrappySolver_srun/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jyang872@gatech.edu

# ---- environment: this job needs python with pyomo/omlt-free base (numpy, pyomo) and the BARON executable + license ----
# The modules in your CrappySolver script (openmpi, metis, openblas, boost, cplex, gurobi) are for the C++ build and are not needed here.
# Fill in the python env and BARON location you use on PACE:
# module load anaconda3 && conda activate pyomo
# export PATH=/path/to/baron:$PATH      # and set BARON_EXE / BARON_LIC in sample_boxes.py

# submit from the repo root (where sample_boxes.py lives), e.g.:  cd /storage/.../decomp && sbatch sample_boxes.sbatch
conda activate pyomo-binder
cd /storage/project/r-jscott319-0/jyang872/decomp
python sample_boxes.py ${SLURM_ARRAY_TASK_ID}
