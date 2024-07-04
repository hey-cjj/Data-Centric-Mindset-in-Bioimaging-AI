export CUDA_VISIBLE_DEVICES=0
python ./TransUNet/test.py \
--dataset Vessel \
--vit_name R50-ViT-B_16 \
--root_path ./data/core \
--num_classes 2 \
--img_size 224 \
--n_skip 3 \
--batch_size 2 \
--max_epochs 200 \
--is_savenii \
--ckpt ./TransUNet/model/TU_Vessel224_critical/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs2_lr0.0001_224/epoch_199.pth \
--test_save_dir ./TransUNet/prediction/test