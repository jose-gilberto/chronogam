import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Tuple


class BaseDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.from_numpy(self.x[index]).float()
        label = torch.from_numpy(np.array(self.y[index]))
        return sample, label
    
