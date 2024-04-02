import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import Dataset

class AntMazeDiffusionDataset(Dataset):
    def __init__(self, data, H=20):
        self.states_mean = data['observations'].mean(1)
        self.states_std = data['observations'].std(1)
        self.states = (data['observations']-self.states_mean[:, None])/self.states_std[:, None]
        self.actions = data['actions']
        self.timeouts = np.where(data['timeouts']==True)[0]
        self.H = H

    def __len__(self):
        return len(self.timeouts)

    def __getitem__(self, idx):
        idx = self.timeouts[idx]
        t0 = idx - 1000
        t1 = np.random.randint(0, 500)
        t_dif = 1000 - t1
        t2 = np.random.randint(200, t_dif)
        t_jump = t2 // self.H
        states = self.states[t0+t1: t0+t1+t2: t_jump]
        t1 = np.random.randint(0, 900)
        t2 = np.random.randint(5, 100)
        s0 = self.states[t0+t1]
        sg = self.states[t0+t1+t2]
        action = self.actions[t0+t1]
        return states[:self.H], s0, sg, action