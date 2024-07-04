export CUDA_VISIBLE_DEVICES=0
python ./TransUNet/critical_select.py \
--dataset Vessel \
--vit_name R50-ViT-B_16 \
--root_path ./data/train \
--num_classes 2 \
--img_size 224 \
--n_skip 3 \
--ckpt ./TransUNet/model/TU_Vessel224_critical/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs2_lr0.0001_224/epoch_199.pth \
--dropout 0.1 \
--num_samples 30 \
--num_critical 50