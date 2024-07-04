import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torchvision.transforms as transforms
import shutil
import cv2
import tifffile as tf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default=None, help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')
parser.add_argument('--save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint of trained model')
parser.add_argument('--is_pretrain', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    args.is_pretrain = True


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    state_dict = {}
    for key,value in torch.load(args.ckpt).items():
        state_dict[key.split('module.')[-1]] = value
    net.load_state_dict(state_dict)
    net.transformer.attention_dropout_rate = 0.1

    # Inference volume
    net.eval()  
    file_list = os.listdir(args.root_path) 

    if args.list_dir:
        with open(args.list_dir, 'r',) as file:
            lines = file.readlines()
        file_list = [line.strip() for line in lines]

    path = args.root_path
    path_d = args.save_dir

    for i_batch, file_name in tqdm(enumerate(file_list)):
        case_name = file_name
        image_path = os.path.join(args.root_path,case_name)
        image = tf.imread(image_path).astype(np.float32) 
        image /= 65535.
        image = (image - 0.094) / 0.049
        deep, height, width = image.shape
        stride = 50
        
        prediction = torch.zeros((deep,height, width)).float().cuda()

        with torch.no_grad():
            for z in range(deep):
                for y in range(0, height - args.img_size + 1, stride):
                    for x in range(0, width - args.img_size + 1, stride):
                        patch = image[z,y:y+args.img_size, x:x+args.img_size]

                        patch = torch.from_numpy(patch)
                        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0).unsqueeze(0)

                        result = net(patch.cuda()).squeeze() 
                        result = torch.argmax(result,dim=0)
                        prediction[z,y:y+args.img_size, x:x+args.img_size] += result


        prediction = (prediction*255).cpu().numpy().astype(np.uint8)
        shutil.copy(os.path.join(path,case_name),os.path.join(path_d,'image',case_name))
        tf.imwrite(os.path.join(path_d,'pred',case_name),prediction)