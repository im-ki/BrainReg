import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pickle
import torch.nn.functional as F

from autolm import AutoLM
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


dir_checkpoint = 'checkpoints/'

class LossFunc(nn.Module):
    def __init__(self, weight, device):
        super(LossFunc, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, points, true):
        error = points - true
        error = torch.sum(torch.pow(error,2))
        return error

class smooth(nn.Module):
    def __init__(self, weight, device):
        super(smooth, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, points):
        error = points[:, :-1, :] - points[:, 1:, :]
        error = torch.sum(torch.pow(error,2))
        return error/100

class Gen_data(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx, :, :]
        label = self.labels[idx, :, :]
        return img, label



def train_net(net,
              device,
              epochs=5,
              batch_size=10,
              weight = 0,
              lr=0.001,
              save_cp=True):
    with open('data.pkl', 'rb') as f:
        all_img, all_lm= pickle.load(f)

    dataset = Gen_data(all_img, all_lm)


    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    writer = SummaryWriter(comment='LR_{}_BS_{}'.format(lr, batch_size))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
        Weight:          {}
    '''.format(epochs, batch_size, lr, save_cp, device.type, weight))

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0)
    lambda1 = lambda epoch: 0.95 ** (epoch/80.)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    criterion1 = LossFunc(weight, device)
    criterion2 = smooth(weight, device)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        sample_cnt = 0
        det_sum = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs, lms = batch

                imgs = imgs.to(device=device, dtype=torch.float32)
                lms = lms.to(device=device, dtype=torch.float32)
                
                outs = net(imgs)
                loss1 = criterion1(outs, lms)
                loss2 = criterion2(outs)

                loss = loss1 + loss2
                epoch_loss += loss.item()

                writer.add_scalar('Loss/rigid_loss', loss1.item(), global_step)

                sample_cnt += 1
                pbar.set_postfix(**{'loss(batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt})

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
            if (epoch+1) % 1 == 0:
                torch.save(net.state_dict(),
                           dir_checkpoint + 'CP_epoch{}.pth'.format(epoch + 1))
                logging.info('Checkpoint {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the BSNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=1,
                        help='The weight of the custom loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    net = AutoLM(device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  weight = args.weight,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
