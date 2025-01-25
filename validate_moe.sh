#!/bin/bash
NUM_PROC=1
DATA_PATH="./imagenet"
checkpoint=./output/train/20250124-172338-gc_vimoe_base-224/model_best.pth.tar
BS=8

python validate_moe.py --model gc_vimoe_base --config ./output/train/20250124-172338-gc_vimoe_base-224/args.yaml --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS
