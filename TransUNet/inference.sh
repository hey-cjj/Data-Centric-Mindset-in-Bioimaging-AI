python ./TransUNet/inference.py \
--dataset Vessel \
--vit_name R50-ViT-B_16 \
--root_path ./data/origin \
--num_classes 2 \
--img_size 224 \
--n_skip 3 \
--ckpt ./TransUNet/model/TU_Vessel224_critical/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs2_lr0.0001_224/epoch_199.pth \
--save_dir ./TransUNet/prediction/inference\
--list_dir ./TransUNet/lists/inference_list.txt 