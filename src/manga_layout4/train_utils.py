from __future__ import annotations
import copy, math, random
import numpy as np
import torch


def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self,model,decay:float):
        self.decay=decay; self.model=copy.deepcopy(model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        msd=model.state_dict(); esd=self.model.state_dict()
        for k,v in esd.items():
            src=msd[k].detach()
            if v.dtype.is_floating_point: v.mul_(self.decay).add_(src,alpha=1-self.decay)
            else: v.copy_(src)


def lr_at(step:int,total:int,warmup:int,base:float,min_ratio:float):
    if step<warmup: return base*max(1,step+1)/max(1,warmup)
    p=min(1.0,max(0.0,(step-warmup)/max(1,total-warmup)))
    return base*(min_ratio+(1-min_ratio)*0.5*(1+math.cos(math.pi*p)))
