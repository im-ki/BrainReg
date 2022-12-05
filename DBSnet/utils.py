import os
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from PIL import Image
import matplotlib.pyplot as plt
import time
import torch

def bc_metric(mapping):
    N, C, H, W = mapping.shape
    device = mapping.device
    face, vertex = image_meshgen(H, W)
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]
    
    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]
    
    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi
    
    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]
    
    ti = mapping[:, face[:, 0], 1]
    tj = mapping[:, face[:, 1], 1]
    tk = mapping[:, face[:, 2], 1]
    
    sjsi = sj - si
    sksi = sk - si
    tjti = tj - ti
    tkti = tk - ti
    
    a = (sjsi * hkhi + sksi * hihj) / area / 2;
    b = (sjsi * gigk + sksi * gjgi) / area / 2;
    c = (tjti * hkhi + tkti * hihj) / area / 2;
    d = (tjti * gigk + tkti * gjgi) / area / 2;
    
    down = (a+d)**2 + (c-b)**2 + 1e-8
    up_real = (a**2 - d**2 + c**2 - b**2)
    up_imag = 2*(a*b+c*d)
    real = up_real / down
    imag = up_imag / down

    mu = torch.stack((real, imag), dim=1)
    return mu

def detD_loss(mapping):
    N, C, H, W = mapping.shape
    device = mapping.device
    face, vertex = image_meshgen(H, W)
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]
    
    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]
    
    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi
    
    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]
    
    ti = mapping[:, face[:, 0], 1]
    tj = mapping[:, face[:, 1], 1]
    tk = mapping[:, face[:, 2], 1]
    
    sjsi = sj - si
    sksi = sk - si
    tjti = tj - ti
    tkti = tk - ti
    
    a = (sjsi * hkhi + sksi * hihj) / area / 2;
    b = (sjsi * gigk + sksi * gjgi) / area / 2;
    c = (tjti * hkhi + tkti * hihj) / area / 2;
    d = (tjti * gigk + tkti * gjgi) / area / 2;
    
    loss = torch.mean(torch.nn.ReLU()(-(a*d-c*b)))
    return loss #mu

def coo_iv2torch(ind, val, N_vertex):
    rmx = np.arange(1, N_vertex)
    rmy = np.arange(1, N_vertex)
    ind, val = ind.numpy(), val.numpy()

    A = sps.csr_matrix((val, ind), shape=(N_vertex, N_vertex))
    Ax, Ay = A[rmx].tocoo(), A[rmy].tocoo()
    Ax_val, Ay_val = Ax.data, Ay.data
    Ax_ind, Ay_ind = np.vstack((Ax.row, Ax.col)), np.vstack((Ay.row, Ay.col))

    ind_x = torch.LongTensor(Ax_ind)
    val_x = torch.FloatTensor(Ax_val)
    Ax = torch.sparse.FloatTensor(ind_x, val_x, torch.Size((rmx.shape[0], N_vertex)))

    ind_y = torch.LongTensor(Ay_ind)
    val_y = torch.FloatTensor(Ay_val)
    Ay = torch.sparse.FloatTensor(ind_y, val_y, torch.Size((rmy.shape[0], N_vertex)))
    return [Ax, Ay]

