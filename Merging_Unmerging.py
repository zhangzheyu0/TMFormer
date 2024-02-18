'''
Created by zhangzy
From  TMFormer: Token Merging Transformer for Brain Tumor Segmentation with Missing Modalities
'''
from typing import Callable, Tuple
import torch
import torch.nn as nn

def do_nothing(x, mode=None):
    return x

### merging process in the Uni-modal Merging Block
def region_matching_random3d(
    metric: torch.Tensor, # The feature for calculating similarity.
    region: torch.Tensor, # The coarse segmentation mask.
    num: int, # The sampled tokens in each window.
    d: int, h: int, w: int, # The initial 3D shape.
    sz:int, sy: int, sx: int, # The 3D shape of window.
    r: int, # The number of decreased tokens. r=d*h*w-N1
) -> Tuple[Callable, Callable]:
    
    B, t, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing
    # 1. Partition
    def rand_sample_on_region(rangesize, shape, region, device):
        sz, sy, sx = rangesize
        dsz, hsy, wsx, num = shape

        range_each  = int(sz*sy*sx//num)
        rand_idx = torch.zeros(shape).to(device)

        rand_idx_num_foreground = torch.zeros((dsz, hsy, wsx, num), device=device).to(torch.int64)
        rand_idx_num_background = torch.zeros((dsz, hsy, wsx, 1), device=device).to(torch.int64)

        for i in range(num):
            rand_idx_num_foreground[:,:,:,i] = torch.randint(low=range_each*i, high=range_each*(i+1), size=(dsz, hsy, wsx))

        rand_idx_num_background[:,:,:,0] = torch.randint(low=0, high=range_each*num, size=(dsz, hsy, wsx))

        rand_idx = torch.where(region[..., None], rand_idx_num_foreground, rand_idx_num_background)

        return rand_idx

    with torch.no_grad():
        dsz, hsy, wsx = d//sz, h // sy, w // sx 
        if num==1:
            rand_idx = torch.randint(sz*sy*sx, size=(dsz, hsy, wsx, num)).to(metric.device)
        else:
            rand_idx = rand_sample_on_region(rangesize=[sz, sy, sx], shape=[dsz, hsy, wsx, num], region=region, device=metric.device)
        
        idx_buffer_view = torch.zeros(dsz, hsy, wsx, sz*sy*sx, device=metric.device, dtype=torch.int64) 
        idx_buffer_view.scatter_(dim=-1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype)) # Set value who is corresponding to rand_idx with -1
        idx_buffer_view = idx_buffer_view.view(dsz, hsy, wsx, sz, sy, sx).permute(0,3,1,4,2,5).reshape(dsz*sz, hsy*sy, wsx*sx) 

        num_B = torch.abs(torch.sum(idx_buffer_view)) # The total token count of Set B.
        rand_idx = idx_buffer_view.reshape(1, -1, 1).argsort(dim=1) 

        del idx_buffer_view

        # 2. Sampling tokens based on region && 3. Putting the sampled tokens into B and the rest tokens are in A
        a_idx = rand_idx[:, num_B:, :] # Set A
        b_idx = rand_idx[:, :num_B, :] # Set B

        def split(x):
            C = x.shape[-1]
            SetA = torch.gather(x, dim=1, index=a_idx.expand(B, t - num_B, C))
            SetB = torch.gather(x, dim=1, index=b_idx.expand(B, num_B, C))
            return SetA, SetB

        metric = metric / metric.norm(dim=-1, keepdim=True)
        SetA, SetB = split(metric)
        SetA = SetA.to(torch.float16)
        SetB = SetB.to(torch.float16)

        # 4. Calculating the similarity scores between A and B
        scores = SetA @ SetB.transpose(-1, -2) 
        r = min(SetA.shape[1], r)

        # 5. Selecting the most similar token in set B for each token in set A
        node_max, node_idx = scores.max(dim=-1) 

        del scores
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 

        SetA_unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens in Set A
        SetA_merge_idx = edge_idx[..., :r, :]  # To be merged Tokens in Set A
        SetB_idx = torch.gather(node_idx[..., None], dim=-2, index=SetA_merge_idx)

    # 6. Merging the most similar token
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        SetA, SetB = split(x)
        n, t1, c = SetA.shape
        
        SetA_unmerge = torch.gather(SetA, dim=-2, index=SetA_unm_idx.expand(n, t1 - r, c))
        SetA_merge = torch.gather(SetA, dim=-2, index=SetA_merge_idx.expand(n, r, c))
        SetB = SetB.scatter_reduce(-2, SetB_idx.expand(n, r, c), SetA_merge, reduce=mode)

        # 7. Appending the dissimilar tokens of Set A
        return torch.cat([SetA_unmerge, SetB], dim=1)

    # Unmerging back
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = SetA_unm_idx.shape[1]
        SetA_unmerge, SetB = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = SetA_unmerge.shape

        SetA_merge = torch.gather(SetB, dim=-2, index=SetB_idx.expand(B, r, c))

        out = torch.zeros(B, t, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_B, c), src=SetB)
        out.scatter_(dim=-2, index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=SetA_unm_idx).expand(B, unm_len, c), src=SetA_unmerge)
        out.scatter_(dim=-2, index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=SetA_merge_idx).expand(B, r, c), src=SetA_merge)

        return out

    return merge, unmerge    




### merging process in the Multi-modal Merging Block
def mod_matching(
    metric: torch.Tensor, # The feature for calculating similarity. 1. It has been constructed in order.
    mask: torch.Tensor, # The modality code.
    r: int,
) -> Tuple[Callable, Callable]:

    t = metric.shape[1]
    r = min(r, t // 2)

    # 2. Partition basd on Equation 5
    def partion_mod(input: torch.Tensor, mask: torch.Tensor):
        N_each = int(input.shape[1]//avaliable_mod)
        if avaliable_mod>=3:
            b = input[:,:2*N_each,:]
            a = input[:,2*N_each:,:]
            b_len = 2*N_each
        elif avaliable_mod==2:
            b = input[:,:N_each,:]
            a = input[:,N_each:,:]
            b_len = N_each
        else:
            b = input
            a = None
            nonlocal r 
            r = 0

        return a, b, b_len

    if r <= 0:
        return do_nothing, do_nothing
    
    avaliable_mod = torch.sum(mask[0])



    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        SetA, SetB, b_len = partion_mod(metric, mask)
        SetA = SetA.to(torch.float16)
        SetB = SetB.to(torch.float16)

        # 3. Calculating the similarity scores between A and B
        scores = SetA @ SetB.transpose(-1, -2) 

        # 4. Selecting the most similar token in set B for each token in set A
        node_max, node_idx = scores.max(dim=-1) 

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] 

        SetA_unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens in Set A
        SetA_merge_idx = edge_idx[..., :r, :]  # To be merged Tokens in Set A
        SetB_idx = torch.gather(node_idx[..., None], dim=-2, index=SetA_merge_idx)

    # 5. Merging the most similar token
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        SetA, SetB, _ = partion_mod(x, mask)
        n, t1, c = SetA.shape
        SetA_unmerge = SetA.gather(dim=-2, index=SetA_unm_idx.expand(n, t1 - r, c)) 
        SetA_merge = SetA.gather(dim=-2, index=SetA_merge_idx.expand(n, r, c)) 
        SetB = SetB.scatter_reduce(-2, SetB_idx.expand(n, r, c), SetA_merge, reduce=mode) 

        # 6. Appending the dissimilar tokens of Set A
        return torch.cat([SetA_unmerge, SetB], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = SetA_unm_idx.shape[1]
        SetA_unmerge, SetB = x[..., :unm_len, :], x[..., unm_len:, :] 
        n, _, c = SetA_unmerge.shape 

        SetA_merge = SetB.gather(dim=-2, index=SetB_idx.expand(n, r, c)) 

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., :b_len, :] = SetB

        out.scatter_(dim=-2, index=(b_len + SetA_unm_idx).expand(n, unm_len, c), src=SetA_unmerge) 
        out.scatter_(dim=-2, index=(b_len + SetA_merge_idx).expand(n, r, c), src=SetA_merge) 

        return out

    return merge, unmerge