python -m torch.distributed.launch --nproc_per_node 1 --master_port 11223  train.py --config ./configs/gc_vit_tiny.yml \
    --data_dir ./imagenet \
    --batch-size 2 \
    --amp \
    --tag "mino_playground" \
    --model-ema