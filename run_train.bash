#!/bin/bash
nvidia-smi
CAT=chair
CUDA_VISIBLE_DEVICES=0 python3 train_multi_gpu_pc_compare_dist.py --log_dir 'log/test2_' --num_point 64 --category $CAT --learning_rate_dpdist 0.0001
