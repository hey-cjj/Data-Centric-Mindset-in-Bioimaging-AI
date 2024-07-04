import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import os

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        miou=dice/(2-dice)
        return dice, hd95,miou
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 0
    else:
        return 0, 0, 0

def test_single_volume(args, image, label,  net, 
                       test_save_path=None, case=None):
    classes = args.num_classes
    net.cuda()
    net.eval()
    inputs = image
    inputs = image.cuda()
    label = label.numpy()
    image = image.numpy()
    with torch.no_grad():
        out = net(inputs)
        out = torch.argmax(torch.softmax(out, dim = 1), dim = 1)
        prediction = out.cpu().detach().numpy()
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        cv2.imwrite(os.path.join(test_save_path, str(case) + "_img.png"),(np.transpose(image[0], (1,2,0))*255).astype(np.uint8))
        cv2.imwrite(os.path.join(test_save_path, str(case) + "_pred.png"),(prediction[0,:,:]*255).astype(np.uint8))
        cv2.imwrite(os.path.join(test_save_path, str(case) + "_gt.png"),(label*255).astype(np.uint8))
    return metric_list