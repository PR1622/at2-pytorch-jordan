#!/bin/bash
set -e

# Multi-GPU on 1 node (example with 2 GPUs)
torchrun --standalone --nproc_per_node=2 train.py --batch_size=64 --epochs=10

# Single GPU debug
# python train.py --batch_size=64 --epochs=10 --save_every=2 --compile=False
