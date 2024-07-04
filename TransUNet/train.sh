export CUDA_VISIBLE_DEVICES=0,1
python ./TransUNet/train.py \
--dataset Vessel \
--img_size 224 \
--vit_name R50-ViT-B_16 \
--root_path ./data/core \
--num_classes 2 \
--max_epochs 200 \
--batch_size 2 \
--n_gpu 2 \
--base_lr 0.0001 \
--n_skip 3 \
--is_pretrain \
--pretrained_path './mae/output/checkpoint-399.pth' \