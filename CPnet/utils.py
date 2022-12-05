import os
import numpy as np
import torch

def mapping2grad_mu(face, vertex, Ind, mask, device):
    """
    Inputs:
        dataset:
            face: (num_of_face, 3) 3 is the index of vertex
            vertex: (num_of_vertex, 2) the location of each vertex
            Ind: (num of vertex, max adjacent)
            mask: (num of vertex, max adjacent)
    Outputs:
        the function to compute the gradient of mu given the mappings
        
    """
    Ind = Ind.to(device=device)
    mask = mask.to(device=device)
    N_adjacent = torch.sum(mask, dim=1).reshape((1, 1, -1))
    mask = mask.reshape((1, 1, mask.shape[0], mask.shape[1]))

    if isinstance(face, np.ndarray):
        face = torch.from_numpy(face.astype(np.int64))
    if isinstance(vertex, np.ndarray):
        vertex = torch.from_numpy(vertex)

    face = face.to(device=device)
    vertex = vertex.to(device=device)

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

    def grad(mapping):
        mapping = torch.transpose(mapping, 1, 2)
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

        return a, b, c, d # shape: (N, num of faces)

    def grad2mu(a, b, c, d):
        down = (a + d) ** 2 + (c - b) ** 2 + 1e-8
        up_real = (a ** 2 - d ** 2 + c ** 2 - b ** 2)
        up_imag = 2 * (a * b + c * d)
        real = up_real / down
        imag = up_imag / down

        mu = torch.stack((real, imag), dim=1)
        return mu

    def compute_dual_mu(mu):
        """
        Inputs:
            mu: m x 1 torch array
            self.Ind: n x max_adjacent, torch array
            self.mask: n x max_adjacent, torch array
        Outputs:
            dual_mu: n x 1 torch array
        """    
        Nv, max_adj = Ind.shape
        N = mu.shape[0]
        mu_adj = mu[:, :, Ind.reshape(-1)].reshape((N, 2, Nv, max_adj)) * mask
        dual_mu = torch.sum(mu_adj, dim=3) / N_adjacent
        return dual_mu

    def compute_mapping2grad_mu(mapping):
        """
        Inputs:
            mapping: (N, 2, num_of_vertex)
        Outputs:
            mu: (N, 2, num_of_face)
            a, b, c, d: (N, 2, num of vertex)
        """
        a, b, c, d = grad(mapping)
        mu = grad2mu(a, b, c, d)
        dual_mu = compute_dual_mu(mu)
        a, b, c, d = grad(dual_mu)
        return mu, a, b, c, d
 
    return compute_mapping2grad_mu


