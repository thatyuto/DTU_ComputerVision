#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J p2_train
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

module load python3/3.10.16
source /zhome/13/2/232841/idlcv_hpc/.venv/bin/activate

# Put this folder on HPC (scp the .py files). Change DATA_ROOT to the Learn
# dataset (4.1) or the no-leakage copy on the cluster (4.2).
cd /zhome/13/2/232841/idlcv_hpc/project2

DATA_ROOT="/dtu/datasets1/02516/ucf101noleakage"

# Debug first: --epochs 1 --max-batches 2
# Then one job per model: per_frame | late | early | 3d | dual
python train.py \
  --data-root "$DATA_ROOT" \
  --model per_frame \
  --epochs 20 \
  --batch-size 8 \
  --out-dir runs/per_frame
