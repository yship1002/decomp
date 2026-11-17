#!/bin/bash
#SBATCH -Jed_train
#SBATCH --array=0-2                       # task 0,1,2 -> scenario 1,2,3
#SBATCH --account=gts-jscott319
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH -t00:30:00                        # a 4x4 network on ~50 rows trains in seconds to a few minutes on CPU
#SBATCH -qinferno
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-jscott319-0/jyang872/CrappySolver_srun/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jyang872@gatech.edu

# Submit AFTER the sampling array has finished, or chain it:
#   JOB=$(sbatch --parsable sample_boxes.sbatch) && sbatch --dependency=afterany:$JOB train_nn.sbatch
# Needs torch (CPU is enough), onnx, onnxruntime, scikit-learn, numpy. Training does not need pyomo/BARON.
module load pytorch/2.1.0           # check once interactively:  python -c "import torch, onnx, onnxruntime, sklearn, numpy"
                                    #   missing ones:  pip install --user onnx onnxruntime scikit-learn "numpy<2"   (torch 2.1 needs numpy 1.x)
which python; python --version
python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
# alternative: use your conda env instead of the module (then it must contain all five packages):
conda activate pyomo-binder
cd /storage/project/r-jscott319-0/jyang872/decomp
python train_nn.py ${SLURM_ARRAY_TASK_ID}
