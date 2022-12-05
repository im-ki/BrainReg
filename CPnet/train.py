import argparse
import logging
import os
import sys

import numpy as np
import torch
import pickle
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from LMnet import LMnet
from bsnet import BSNet
from utils import mapping2grad_mu

from torch.utils.data import DataLoader, Dataset

size = [128, 128]
dir_checkpoint = 'checkpoints/'

class Mu_Loss(nn.Module):
    def __init__(self, weight, device, mapping2grad_mu):
        super(Mu_Loss, self).__init__()
        self.weight = weight
        self.m2g = mapping2grad_mu 
        
    def forward(self, mapping):
        N = mapping.shape[0]
        mu, a, b, c, d = self.m2g(mapping)
        norm_sqr = a**2 + b**2 + c**2 + d**2
        N_face = norm_sqr.shape[1]
        return (torch.sum(norm_sqr) + torch.sum(mu**2)) / N / N_face, torch.sum(norm_sqr) / N / N_face, torch.sum(mu**2) / N / N_face

class LM_Loss(nn.Module):
    def __init__(self, weight, device):
        super(LM_Loss, self).__init__()
        self.weight = weight
        self.device = device
    def forward(self, mapping, lm, landmarks):
        N = mapping.shape[0]
        num_of_vertex = mapping.shape[2]
        mapping_trans = torch.transpose(mapping, 1, 2) 
        mapping_inline = mapping_trans.reshape( N*num_of_vertex, 2)

        bias = torch.arange(N).reshape((N,1))* num_of_vertex
        bias = bias.to(device=self.device)

        LM_after = mapping_inline[lm+bias, :]
        landmarks = torch.unsqueeze(landmarks, 0)
        LM_diff = LM_after - landmarks
        LM_diff_sum_pow = torch.sum(torch.pow(LM_diff,2))/N
 
        return LM_diff_sum_pow

def get_training_data(mapping, lm, device):
    N = mapping.shape[0]
    num_of_vertex = mapping.shape[2]
    mapping_trans = torch.transpose(mapping, 1, 2)
    mapping_inline = mapping_trans.reshape(N*num_of_vertex, 2)

    bias = torch.arange(N).reshape((N,1))* num_of_vertex
    bias = bias.to(device=device)

    LM_after = mapping_inline[lm+bias, :]
    return torch.transpose(LM_after, 1, 2)


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


def train_net(bsnet_load,
              lnet,
              device,
              epochs=5,
              batch_size=1,
              weight = 0,
              lr=0.001,
              save_cp=True):

    with open('./data.pkl', 'rb') as f:
        imgs, As, ind_s2d, bound_num, vertex, face, Ind, mask, lm = pickle.load(f)

    dataset = Gen(imgs, As)
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    bsnet = BSNet(size, device, vertex, bound_num, n_channels=2, n_classes=2, bilinear=True)
    bsnet.to(device=device)

    if bsnet_load:
        bsnet.load_state_dict(
            torch.load(bsnet_load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(bsnet_load))
    else:
        print('Need a pretrained BSNet')
        sys.exit(0)

    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
    '''.format(epochs, batch_size, lr, save_cp, device.type))

    optimizer = optim.RMSprop(lnet.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    
    criterion1 = LM_Loss(weight, device)
    criterion2 = Mu_Loss(weight, device, mapping2grad_mu(face, vertex, Ind, mask, device))

    landmarks = torch.from_numpy(vertex[lm, :]).to(device=device)
    lm = lm.to(device=device)

    mesh = torch.from_numpy(vertex.astype(np.float32)).unsqueeze(0).to(device=device) 
    for epoch in range(epochs):
        lnet.train()
        bsnet.eval()

        epoch_loss = 0
        sample_cnt = 0
        sum_detD = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
                mu,_ = batch

                h, w = mu.shape[2], mu.shape[3]

                mu = mu.to(device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    mapping_standard = bsnet(mu)[0] 
                    sampled_pts_raw = get_training_data(mapping_standard, lm.unsqueeze(0).repeat(batch_size, 1), device) 
                    N = sampled_pts_raw.shape[0]

                    N_sel = 64            
                    N_vt = mapping_standard.shape[2]
                    spaned_mesh = torch.transpose(mesh, 1, 2).repeat(1, 1, N_sel)
                    sampled_pts_raw = sampled_pts_raw.reshape((N, 2, N_sel, 1)).repeat(1, 1, 1, N_vt).reshape((N, 2, N_vt*N_sel))
                    dis = torch.sum((spaned_mesh - sampled_pts_raw)**2, dim = 1)
                    dis = dis.reshape((N, N_sel, -1))
                    argmin_dis = torch.argmin(dis, dim=2)
                    sampled_pts = get_training_data(spaned_mesh.repeat(N, 1, 1), argmin_dis, device)

                mu_recon = lnet(sampled_pts.reshape((batch_size, 1, -1)))
                map_pred = bsnet(mu_recon)[0] 

                loss1 = criterion1(map_pred, argmin_dis, landmarks) 
                loss2, grad_mu, mu_sqr = criterion2(map_pred)
                loss = 1000*loss1 + loss2
                epoch_loss += loss.item()
                sample_cnt += 1

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt, 'Intensity loss:': loss1.item(), 'mu loss': loss2.item(), 'grad_mu': grad_mu.item(), 'mu_sqr': mu_sqr.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(lnet.parameters(), 0.1)
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(lnet.state_dict(),
                       dir_checkpoint + 'CP_epoch{}.pth'.format(epoch + 1))
            logging.info('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00002,
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
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    lnet = LMnet(device)
    lnet.to(device=device)

    try:
        train_net(bsnet_load = args.load,
                  lnet=lnet,
                  epochs=args.epochs,
                  weight = args.weight,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(lnet.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
