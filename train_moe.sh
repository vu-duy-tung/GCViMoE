export CUDA_VISIBLE_DEVICES=1,2
export TORCH_DISTRIBUTED_DEBUG=OFF

python -m torch.distributed.launch --nproc_per_node 2 --master_port 11223  train_moe.py --config ./configs/gc_vimoe_base.yml \
    --data_dir ./imagenet \
    --batch-size 8 \
    --tag "mino_playground"