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
from datasets.dataset_vessel import Vessel_Uncertainty
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torchvision.transforms as transforms
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint of trained model')
parser.add_argument('--is_pretrain', action='store_true')
parser.add_argument('--dropout', type=float,  default=0.1, help='dropout rate in vit block')
parser.add_argument('--num_samples', type=int,  default=50, help='number of samples in MC dropout')
parser.add_argument('--num_critical', type=int,  default=10, help='number of critical data')

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

    dataset_config = {
        'Vessel': {
            'Dataset': Vessel_Uncertainty,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.is_pretrain = True


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    config_vit.transformer.attention_dropout_rate = args.dropout
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    state_dict = {}
    for key,value in torch.load(args.ckpt).items():
        state_dict[key.split('module.')[-1]] = value
    net.load_state_dict(state_dict)

    db_test = Vessel_Uncertainty(args.root_path,'test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    net.train()  # dropout

    U = {}
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch
        case_name = case_name[0]
        h, w = image.size()[2:]
        predictions = []
        for i in range(args.num_samples):
            with torch.no_grad():
                output = net(image.cuda()).squeeze()
                predictions.append(torch.nn.functional.softmax(output, dim=0))
        predictions = torch.stack(predictions,dim=0)
        variance = torch.var(predictions, dim=0)
        uncertainty = torch.sum(variance)
        U[case_name] = uncertainty
    
    sorted_U = sorted(U, key=U.get, reverse=True)
    
    with open('uncertainty.txt', 'w') as f:
        for case_name in sorted_U:
            f.write(case_name+' '+f"{U[case_name]:.4f}"+'\n')
        f.close()   
    with open('critical.txt', 'w') as f:
        critical = sorted_U[:args.num_critical]
        for case_name in critical:
            f.write(case_name+'\n')
        f.close()  