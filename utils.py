import numpy as np
from scipy import sparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



def convert_genome_to_bin(reg, res):
    r = []
    for rr in range(4):
        r.append(int(reg[rr]/res))
        
    return r


def get_mat_data(hic, data_root, cell_line, event, chrom_nm, indices, res, win_start, win_end):

    matrix_object = hic.getMatrixZoomData(chrom_nm, chrom_nm, "observed", "NONE", "BP", res)
    region = [win_start, win_end, win_start, win_end]
    origin_end = convert_genome_to_bin(region, res)

    orig_r = origin_end[0]
    end_r = origin_end[1]
    orig_c = origin_end[2]
    end_c = origin_end[3]

    n_rows = end_r - orig_r + 1
    n_cols = end_c - orig_c + 1

    contact_mat = np.zeros((n_rows, n_cols))
    # print(orig_r, orig_c)

    counter = 0
    for i,cr in enumerate(matrix_object.getRecords(region[0], region[1], region[2], region[3])):
        row = int(cr.binX/res-orig_r)
        col = int(cr.binY/res-orig_c)
        
        contact_mat[row][col] = cr.counts
        contact_mat[col][row] = cr.counts
        
        counter += 1
        
    print("\n\ntotal: ", counter)
    print('Matrix shape: ', contact_mat.shape)
    print('Matrix Density: ', 2*counter/(contact_mat.shape[0]*contact_mat.shape[0]))
    
 
    adj_p = contact_mat[indices-int(win_start/200)]
    num_val = np.count_nonzero(adj_p)
    print("Number of valid connections: ", num_val)

    mat_comp = sparse.csr_matrix(adj_p)
    os.makedirs(f'{data_root}/{cell_line}/{event}_adj_mats/{chrom_nm}/', exist_ok=True)
    sparse.save_npz(f'{data_root}/{cell_line}/{event}_adj_mats/{chrom_nm}/{win_start}_{win_end}.npz', mat_comp)

    print(win_start, ' to ', win_end, ':\n', contact_mat)
    return adj_p




## using a GPU
def get_default_device(gpu='cuda:0'):
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device(gpu)
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    


class combine_data(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, eps, adjs, inds, preds):
        'Initialization'
        self.eps = eps
        self.adjs = adjs
        self.inds = inds
        self.preds = preds
      
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.eps)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ep = self.eps[index]
        adj = self.adjs[index]
        ind = self.inds[index]
        pred = self.preds[index]

        return ep, adj, ind, pred


## preparing dataloader for a single window
def get_dataloader(ep_set, adj_set, ind_set, pred_set, BATCH_SIZE, gpu='cuda:0'):
    
    dataset = combine_data(ep_set, adj_set, ind_set, pred_set)
    device = get_default_device(gpu)

    return DeviceDataLoader(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True), device)



class MLoss(nn.Module):
    def __init__(self, eta1, eta2, eta3):
        """
        Initializes the loss function.
        """
        super(MLoss, self).__init__()
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def forward(self, input, target):
        """
        Defines the forward pass of the loss function.
        input: The predicted values from the model.
        target: The ground truth values.

        Returns the computed loss.
        """
        loss1 = F.l1_loss(input, target)
        loss2 = F.huber_loss(input, target, delta=0.7) 
        loss3 = (torch.var(input)-torch.var(target))**2
        
        return self.eta1*loss1 + self.eta2*loss2 + self.eta3*loss3    

