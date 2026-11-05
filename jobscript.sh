#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J hpc_ex21
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 00:15
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err
module load python3/3.10.16
source /zhome/13/2/232841/idlcv_hpc/.venv/bin/activate
cd /zhome/13/2/232841/idlcv_hpc
python gpu_tensor_demo.py