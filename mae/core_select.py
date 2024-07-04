import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
import cv2 
from dataset import Vessile
from scipy import spatial
from tqdm import tqdm
import shutil
import gc
import time
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('core_select', add_help=False)
	parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
	parser.add_argument('--ckpt', default='./mae/output/checkpoint-399.pth', type=str)
	parser.add_argument('--root_path', default='./data/train', type=str)
	parser.add_argument('--save_path', default='./data/core/image', type=str)
    parser.add_argument('--num_core', default=30, type=int,
                        help='scale of core set')
	return parser

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def cosine_distance_matrix(tensor1, tensor2):
    norm_tensor1 = torch.norm(tensor1, dim=1, keepdim=True)
    norm_tensor2 = torch.norm(tensor2, dim=1, keepdim=True)

    norm_tensor1 = tensor1 / norm_tensor1
    norm_tensor2 = tensor2 / norm_tensor2

    dot_product = 1 - torch.mm(norm_tensor1, norm_tensor2.t())

    return dot_product

def calc_F(distance,options,images_selected,keys):
	max_dis = float('-inf')
	best_option = None
	for option in options:
		min_dis = float('inf')
		for image_selected in images_selected:
			min_dis = min(min_dis,distance[keys[option],keys[image_selected]])

		if max_dis < min_dis:
			max_dis = min_dis
			best_option = option
	return best_option


def main():
	args = get_args_parser()
    args = args.parse_args()
	ckpt_dir = args.ckpt
	model = prepare_model(ckpt_dir, args.model)

	path = args.root_path
	path_d = args.save_path
	if not os.path.exists(path_d):
		os.mkdir(path_d)

	dataset = Vessile(path,'test')
	embeddings = []
	options_available = []
	keys = {}
	model.eval()
	model.cuda()
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,drop_last=False)
	for i, (img, _, name) in tqdm(enumerate(dataloader)):
		embedding, _, _ = model.forward_encoder(img.float().cuda(), mask_ratio=0) # set mask_ratio=0 when select core set
		embeddings.append(embedding.detach().cpu().flatten())
		options_available.append(name[0])
		keys[name[0]] = i
		torch.cuda.empty_cache()

	embeddings = torch.stack(embeddings,dim=0)
	embeddings = embeddings.unsqueeze(1)
	scale_factor = embeddings.shape[2]/1000	       # downsample embeddings
	embeddings = F.interpolate(embeddings, scale_factor=1/scale_factor, mode='linear')
	embeddings = embeddings.squeeze(1)

	print(embeddings.shape)
	print(embeddings.dtype)

	distance = []
	distance = cosine_distance_matrix(embeddings,embeddings)

	np_distance = distance.detach().cpu().numpy()
	options_available = options_available
	keys = {key:keys[key] for key in options_available}

	images_selected = []

	# random initial images_selected
	np.random.seed(4099)
	for i in range(3):
		init_id = np.random.randint(0,len(options_available))
		init_data = options_available[init_id]
		images_selected.append(init_data)
		options_available.remove(init_data)
		shutil.copy(os.path.join(path,init_data),os.path.join(path_d,init_data))

	for i in tqdm(range(args.num_core-len(images_selected))):
		start = time.time()
		best_option = calc_F(distance,options_available,images_selected,keys)
		end = time.time()
		print(best_option)
		
		options_available.remove(best_option)
		images_selected.append(best_option)
		shutil.copy(os.path.join(path,best_option),os.path.join(path_d,best_option))

	for image_selected in images_selected:
		with open('./data/coreset.txt','a') as f:
			f.write(image_selected+'\n')
	print(images_selected)

if __name__=='__main__':
	main()