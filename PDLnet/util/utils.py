import logging
import os

import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from util.model import *
from util.Resnet import *
from util.time_utils import *


def adjust_learning_rate(optimizer, epoch, train_epoch, learning_rate):
    lr = learning_rate * (0.6 ** (epoch / train_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target):
    output = np.array(output.data.cpu().numpy())
    target = np.array(target.data.cpu().numpy())
    pre = 0
    for i in range(output.shape[0]):
        pos = np.unravel_index(np.argmax(output[i]), output.shape)
        pre_label = pos[1]
        if pre_label == target[i]:
            pre += 1
    pre /= target.size
    pre *= 100
    return pre


def loadTrainData(batch_size, train_path):
    logging.info('==> Preparing TrainData..')
    transform_train = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.Resize((260, 260)),
        transforms.RandomCrop(256, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.49, 0.44, 0.39), (0.26, 0.25, 0.24)),
    ])
    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainLoader


# 测试集
def loadTestData(batch_size, test_path):
    logging.info('==> Preparing TestData..')
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize([0.47, 0.43, 0.40], [0.28, 0.27, 0.26]),
    ])
    testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=0)
    return testLoader

