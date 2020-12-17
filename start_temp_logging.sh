#!/bin/bash
proj="temperatuur"
group=$1
name=$2

kill $(ps aux | grep "train" | grep -v grep | awk '{print $2}')

screen -d -m python gpumon.py --wbgroup "$group" --wbname "$name" --wbproj "$proj"

screen -d -m OMP_NUM_THREADS=12 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2740 \
    train.py --ada_length 100000 --augment True --d_skip False --la_alpha 0.8 --la_steps 1000 --lookahead True \
    --lr 0.002 --r1 0.000001 --mixing_prob 0 --path_regularize 0 --iter 15000 --size 1024 --batch_size 2 \
    --wbgroup "$group" --wbname "$name" --wbproj "$proj" --val_batch_size 4 --num_accumulate 4  --eval_every -1 \
    --checkpoint ~/modelzoo/maua-sg2/cyphept-CYPHEPT-2q5b2lk6-39-1024-100000.pt --path ~/trainsets/alaeset

sshpass -p 23023 pi@192.168.1.221 screen -d -m python3 wandb_temp.py --wbgroup "$group" --wbname "$name" --wbproj "$proj"