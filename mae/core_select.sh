export CUDA_VISIBLE_DEVICES=0
python ./mae/core_select.py \
--num_core 30 \
--root_path './data/train' \
--save_path './data/core/image' \
--model 'mae_vit_base_patch16' \
--ckpt './mae/output/checkpoint-399.pth'