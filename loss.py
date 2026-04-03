import torch
import torch.nn as nn
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma
import random
import numpy as np
import torch.nn.functional as F
import math
import collections


def dynamic_thresholds(sim_matrix, labels):
    thresholds = []
    for i in range(len(labels)):
        class_mask = labels == labels[i]
        class_sim = sim_matrix[i][class_mask]
        mean_sim = class_sim.mean()
        std_sim = class_sim.std()
        thresholds.append(mean_sim - std_sim)
    return torch.tensor(thresholds).to(sim_matrix.device)
 

def check_and_correct_image(image):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        return image
    
    
def wavelet_filter(image):
    return denoise_wavelet(image, rescale_sigma=True)

def non_local_means_filter(image):
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    if np.isnan(sigma_est) or sigma_est == 0:
        sigma_est = 0.01
    return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)

def mixup_data(x, y,device,num_classes, alpha=5.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    y = torch.zeros(batch_size, num_classes).to(device).scatter_(
            1, y.view(-1, 1), 1)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    # return mixed_x, y_a, y_b, lam
    return mixed_x,mixed_y,lam

def mixup_criterion(criterion, pred,label, lam):
    return -torch.mean(torch.sum(F.log_softmax(pred,dim=1)*label,dim=1))
    # return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)





