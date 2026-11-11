#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J p3_seg
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

module load python3/3.10.16
source /zhome/13/2/232841/idlcv_hpc/.venv/bin/activate

cd /zhome/13/2/232841/idlcv_hpc/project3

# Debug first: --epochs 1 --max-batches 2
# 任务 2/3: --model encdec | unet，两个数据集各跑一次
# 任务 4:   --loss bce | focal | weighted_bce
python train.py --dataset ph2 --model unet --loss bce --epochs 50
python eval.py --ckpt runs/ph2_unet_bce/best.pt
