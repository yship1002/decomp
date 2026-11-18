#!/bin/bash
#SBATCH -Jed_pipeline
#SBATCH --array=0-2                       # task 0,1,2 -> scenario 1,2,3 (three independent jobs, each runs sample -> train -> certify)
#SBATCH --account=gts-jscott319
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH -t08:00:00                        # sampling <= 64 x 30 s (~32 min) + training (minutes) + certification 600 s + model builds
#SBATCH -qinferno
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-jscott319-0/jyang872/decomp_srun/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jyang872@gatech.edu

# One environment must contain: pyomo, omlt, onnx, numpy, torch (CPU is enough); onnxruntime is optional.
#   once, on a login node:  conda activate pyomo-binder;  python -m pip install omlt onnx torch
# Do NOT `module load pytorch`: it redefines `python` as a container without pyomo.
module load anaconda3               # makes `conda activate` work inside a batch job
conda activate pyomo-binder
which python                        # must print a path inside the pyomo-binder env
python -c "import pyomo, omlt, onnx, numpy, torch; print('imports ok')"
cd /storage/project/r-jscott319-0/jyang872/decomp
python pipeline.py ${SLURM_ARRAY_TASK_ID}
