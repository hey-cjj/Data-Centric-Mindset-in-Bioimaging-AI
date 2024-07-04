import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.dataset_vessel import Vessel
from utils import test_single_volume
from torch.utils.data import Sampler

class sampler(Sampler):
    def __init__(self,data):
        super(sampler,self).__init__(data)
        self.data = data
        self.idx = []
        # number of add_data is 3, and sample 7 samples from origin_data
        indexes = random.sample(range(len(self.data)-3), 7)   
        self.idx = indexes + list(range(len(self.data))[-3:])

    def __iter__(self):
        random.shuffle(self.idx)
        return iter(self.idx)
    
    def __len__(self):
        return len(self.data)


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.add_path:
        db_train = Vessel(args.root_path, 'train', args.add_path)
        trainloader = DataLoader(db_train, batch_size=batch_size, sampler=sampler(db_train), num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    else:
        db_train = Vessel(args.root_path, 'train')
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    print("The length of train set is: {}".format(len(db_train)))
 
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    soft_params = []
    hard_params = []
    for name, param in model.named_parameters():
        if 'position_embeddings' or 'encoder' in name:
            soft_params.append(map(id, param))
        else:
            hard_params.append(map(id, param))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, _ = sampled_batchE
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            label_batch = label_batch.squeeze(1)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        with torch.no_grad():
            model.eval()   
            if (epoch_num + 1) % 10 == 0:
                db_test = Vessel(args.val_path,split='test')
                testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
                logging.info("{} test iterations per epoch".format(len(testloader)))
                metric_list = 0.0
                for i_batch, sampled_batch in tqdm(enumerate(testloader)):
                    image, label, case_name = sampled_batch
                    case_name = case_name[0]
                    h, w = image.size()[2:]
                    metric_i = test_single_volume(args, image, label, model)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_iou = np.mean(metric_list, axis=0)[2]
            
                if best_performance < mean_iou:
                    best_performance = mean_iou
                    save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_miou_'+ 
                                                                    f"{best_performance.item():.4f}" + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save best model to {}".format(save_mode_path))

            if epoch_num >= max_epoch - 1:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                iterator.close()
                break

    writer.close()
    return "Training Finished!"