import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from models import TemporalUnet, IDM

class UNetDiffusion(nn.Module):
    def __init__(self, s_dim, a_dim, H=32, diffusion_steps=50, predict='epsilon', schedule='linear'):
        super(UNetDiffusion, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.unet = TemporalUnet(horizon=H, transition_dim=s_dim)
        self.idm = IDM(s_dim, a_dim)
        self.diffusion_steps = diffusion_steps
        self.predict = predict
        self.schedule = schedule
        
        if self.schedule == 'linear':
            beta1 = 0.02
            beta2 = 1e-4
            beta_t = (beta1 - beta2) * torch.arange(diffusion_steps+1, 0, step=-1, dtype=torch.float32) / (diffusion_steps) + beta2
            
        alpha_t = 1 - torch.flip(beta_t, dims=[0])
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
        self.register_buffer('beta_t', beta_t)
        self.register_buffer('alpha_t', torch.flip(alpha_t, dims=[0]))
        self.register_buffer('log_alpha_t', torch.flip(log_alpha_t, dims=[0]))
        self.register_buffer('alphabar_t', torch.flip(alphabar_t, dims=[0]))
        self.register_buffer('sqrtab', torch.flip(sqrtab, dims=[0]))
        self.register_buffer('oneover_sqrta', torch.flip(oneover_sqrta, dims=[0]))
        self.register_buffer('sqrtmab', torch.flip(sqrtmab, dims=[0]))
        self.register_buffer('mab_over_sqrtmab_inv', torch.flip(mab_over_sqrtmab_inv, dims=[0]))
        
    def compute_diffusion_loss(self, x):
        t_idx = torch.randint(0, self.diffusion_steps, (x.shape[0], 1)).to(x.device)
        t = t_idx.float().squeeze(1)
        epsilon = torch.randn_like(x).to(x.device)
        x_t = self.sqrtab[t_idx].unsqueeze(2) * x + self.sqrtmab[t_idx].unsqueeze(2) * epsilon
        epsilon_pred = self.unet(x_t, t)
        if self.predict == 'epsilon':
            w = torch.minimum(torch.tensor(5)/((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2), torch.tensor(1)) # Min-SNR-gamma weights
            loss = (w.unsqueeze(1) * (epsilon - epsilon_pred) ** 2).mean()
        elif self.predict == 'x0':
            #epsilon_pred = torch.tanh(epsilon_pred)
            w = torch.minimum((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2, torch.tensor(5))
            loss = (w * (x - epsilon_pred) ** 2).mean()
        return loss
    
    def compute_idm_loss(self, s0, sg, a):
        a_pred = self.idm(s0, sg)
        loss = F.mse_loss(a, a_pred)
        return loss
    