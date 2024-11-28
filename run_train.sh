#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.1 \
                --lr_decay_schedule 150 225 \
                --HSKD 1\
                --experiments_dir 'output' \
                --experiments_name 'test3' \
                --classifier_type 'resnet18_dtskd' \
                --data_path '/mnt/dataset/cifar' \
                --data_type 'cifar100' \
                --backbone_weight 3.0 \
                --b1_weight 1.0 \
                --b2_weight 1.0 \
                --b3_weight 1.0 \
                --ce_weight 0.2 \
                --kd_weight 0.8 \
                --coeff_decay 'cos' \
                --cos_max 0.9 \
                --cos_min 0.0 \
                --rank 0 \
                --tsne 0 \
                --world_size 1


