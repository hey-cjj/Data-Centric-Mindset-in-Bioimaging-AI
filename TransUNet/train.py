import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--add_path', type=str,
                    default=None, help='add dir for data')
parser.add_argument('--val_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for val data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--is_pretrain', action='store_true')
parser.add_argument('--pretrained_path', type=str, help='path to ckpt of pretrained model')
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
    args.exp = 'TU_' + dataset_name + str(args.img_size)+'_critical'
    snapshot_path = "./TransUNet/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    if args.is_pretrain:
        weights=torch.load(args.pretrained_path, map_location='cpu')
        new_state_dict = collections.OrderedDict()
        for key, value in weights['model'].items():
            if key.split('.')[0] == 'norm':
                    new_key = 'transformer.encoder.encoder_norm.'+key.split('.')[-1]
                    new_state_dict[new_key] = value
            elif key.split('.')[0] == 'blocks':
                if 'norm1' in key:
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.attention_norm.'+key.split('.')[-1]
                    new_state_dict[new_key] = value
                elif 'norm2' in key:
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.ffn_norm.'+key.split('.')[-1]
                    new_state_dict[new_key] = value
                elif 'fc1' in key:
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.ffn.fc1.'+key.split('.')[-1]
                    new_state_dict[new_key] = value
                elif 'fc2' in key:
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.ffn.fc2.'+key.split('.')[-1]
                    new_state_dict[new_key] = value
                elif 'qkv' in key:
                    if value.ndim == 1:
                        value = value.reshape(3,-1,dim)
                        q,k,v = value[0,:].squeeze(), value[1,:].squeeze(), value[2,:].squeeze()
                    if value.ndim == 2:
                        _, dim = value.shape
                        value = value.reshape(3,-1,dim)
                        q,k,v = value[0,:,:], value[1,:,:], value[2,:,:]
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.attn.query.'+key.split('.')[-1]
                    new_state_dict[new_key] = q
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.attn.key.'+key.split('.')[-1]
                    new_state_dict[new_key] = k
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.attn.value.'+key.split('.')[-1]
                    new_state_dict[new_key] = v
                elif 'proj' in key:
                    new_key = 'transformer.encoder.layer.'+key.split('.')[1]+'.attn.out.'+key.split('.')[-1]
                    new_state_dict[new_key] = value
    
        net.load_state_dict(new_state_dict, strict = False)
        
    trainer = {'Vessel': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)