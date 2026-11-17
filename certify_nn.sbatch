#!/bin/bash
#SBATCH -Jed_cert
#SBATCH --array=0-2                       # task 0,1,2 -> scenario 1,2,3 (three independent jobs)
#SBATCH --account=gts-jscott319
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH -t00:30:00                        # must exceed TIME_LIMIT in certify_nn.py (600 s) plus model build time
#SBATCH -qinferno
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-jscott319-0/jyang872/decomp_srun/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jyang872@gatech.edu

# Submit AFTER train_nn.sbatch has finished, or chain it:
#   JOB=$(sbatch --parsable train_nn.sbatch) && sbatch --dependency=afterok:$JOB certify_nn.sbatch
# Needs pyomo, omlt, onnx, numpy and the BARON executable + license (paths in certify_nn.py). Same environment as sample_boxes.sbatch.
# Do NOT load the pytorch module here: it redefines `python` as a container that has no pyomo.
module load anaconda3               # makes `conda activate` work inside a batch job
conda activate pyomo-binder
which python                        # must print a path inside your pyomo-binder env (not a container function)
python -c "import pyomo, omlt, onnx, numpy; print('imports ok')"
cd /storage/project/r-jscott319-0/jyang872/decomp
python certify_nn.py ${SLURM_ARRAY_TASK_ID}
