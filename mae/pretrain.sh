export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 
python -m torch.distributed.launch --nproc_per_node=2  --master_port=1234 ./mae/main_pretrain.py \
--input_size 224 \
--batch_size 2 \
--model mae_vit_base_patch16 \
--mask_ratio 0.75 \
--output_dir ./mae/output \
--log_dir ./mae/output \
--epochs 400 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 --accum_iter 4 \
--data_path ./data/train \

