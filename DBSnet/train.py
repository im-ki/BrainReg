import argparse
import logging
import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

import pickle
from bsnet import BSNet
from torch.utils.data import Dataset
from utils import coo_iv2torch, bc_metric, detD_loss

from torch.utils.data import DataLoader


print('Begin...')

data_path = 'test.npy'
size = [128, 128]
dir_checkpoint = 'checkpoints/'

class LossFunc(nn.Module):
    def __init__(self, weight, device):
        super(LossFunc, self).__init__()
        self.device = device
        self.weight = weight
        
    def forward(self, pred, A, org_ind, bound_num):
        batch_size, _, Nv = pred.shape
        A[:, math.floor(bound_num/2), :] = 0
        x = pred[:, 0, :]
        y = pred[:, 1, :]

        ind = org_ind.reshape(-1)

        new_x, new_y = x[:, ind].reshape((batch_size, Nv, A.shape[2])), y[:, ind].reshape((batch_size, Nv, A.shape[2]))
        
        bx_hat = torch.sum((A * new_x)[:, 1:, :], dim=2)
        by_hat = torch.sum((A * new_y)[:, 1:, :], dim=2)


        b_hat = torch.stack((bx_hat, by_hat))

        return torch.mean(torch.abs(b_hat))

class BoundLoss(nn.Module):
    def __init__(self, weight, device):
        super(BoundLoss, self).__init__()
        self.device = device
        self.weight = weight
        
    def forward(self, theta, num_of_bounds):
        N = theta.shape[0]
        return torch.sum(1/(1e8 * (theta+1e-6))) / N

class Gen(Dataset):
    def __init__(self, imgs, As):
        self.imgs = imgs
        self.As = As
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = self.imgs[idx, :, :, :]
        A = self.As[idx, :, :]
        return img, A

def train_net(load,
              device,
              epochs=5,
              batch_size=1,
              weight = 0,
              lr=0.001,
              save_cp=True):

    with open('./data.pkl', 'rb') as f:
        imgs, As, ind_s2d, bound_num, vertex = pickle.load(f)
    dataset = Gen(imgs, As)
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    net = BSNet(size, device, vertex, bound_num, n_channels=2, n_classes=2, bilinear=True)
    if load:
        net.load_state_dict(
            torch.load(load, map_location=device)
        )
    net.to(device=device)

    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
        Weight:          {}
    '''.format(epochs, batch_size, lr, save_cp, device.type, weight))

    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-8, momentum=0.9)
    lambda1 = lambda epoch: (1/10000)*epoch+1 
    criterion1 = LossFunc(weight, device)
    criterion2 = BoundLoss(weight, device)

    all_img = torch.zeros((1, 2, 128, 128))
    all_A = torch.zeros((1, 14168, 8))

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        sample_cnt = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs, A = batch
                all_img = torch.cat((all_img, imgs), 0)
                all_A = torch.cat((all_A, A), 0)
                h, w = imgs.shape[2], imgs.shape[3]

                A = A.to(device=device)

                assert imgs.shape[1] == 2, 'Network has been defined with {} input channels, but loaded images have {} channels. Please check that the images are loaded correctly.'.format(net.n_channels, imgs.shape[1])

                imgs = imgs.to(device=device, dtype=torch.float32)
                
                map_pred, theta = net(imgs)
                loss1 = criterion1(map_pred, A, ind_s2d, bound_num)
                loss2 = criterion2(theta, bound_num)
                loss = loss1 + loss2
                epoch_loss += loss.item()
                sample_cnt += 1

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss1': loss1.item(), 'loss2': loss2.item(), 'epoch avg loss:': epoch_loss / sample_cnt})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP_epoch{}.pth'.format(epoch + 1))
            logging.info('Checkpoint {} saved !'.format(epoch + 1))

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0,
                        help='The weight of the custom loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    train_net(load=args.load,
              epochs=args.epochs,
              weight = args.weight,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device)
