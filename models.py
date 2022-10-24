import torch
import torch.nn as nn
import torch.utils.data
import torchvision as tv
import torch.nn.functional as F
import torchvision.models as models

from torch.optim import Adam
from torch.nn import Parameter

from torchvision.datasets import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from torchvision.utils import save_image
from torchvision.datasets.folder import *
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv2d, BCEWithLogitsLoss, DataParallel, AvgPool2d, ModuleList, LeakyReLU, ConvTranspose2d, Embedding

from utils import config

class G_net(nn.Module):
    def __init__(self,in_dim=config.latent_dim):
        super(G_net,self).__init__()
        self.fc1= nn.Sequential(
            nn.Linear(in_dim,config.units[4]),
            nn.ReLU(),
            nn.Linear(config.units[4],config.units[3]*config.fs[0]*config.fs[1]),
            nn.ReLU(),
            nn.BatchNorm1d(config.units[3]*config.fs[0]*config.fs[1])
        )
        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(config.units[3],config.units[2],config.k_size[3],stride=config.strides[3],padding=config.padding[3],output_padding=config.strides[3]//2),
            nn.BatchNorm2d(config.units[2]),
            nn.ReLU()
        )#[64,12,12]
        self.ct2 = nn.Sequential(
            nn.ConvTranspose2d(config.units[2],config.units[1],config.k_size[2],stride=config.strides[2],padding=config.padding[2],output_padding=config.strides[2]//2),
            nn.BatchNorm2d(config.units[1]),
            nn.ReLU()
        )#[32,27,27]
        self.ct3 = nn.Sequential(
            nn.ConvTranspose2d(config.units[1],config.units[0],config.k_size[1],stride=config.strides[1],padding=config.padding[1],output_padding=config.strides[1]//2),
            nn.BatchNorm2d(config.units[0]),
            nn.ReLU()
        )#[3,57,57]
        self.ct4 = nn.Sequential(
            nn.ConvTranspose2d(config.units[0],config.small_image_size[0],config.k_size[0],stride=config.strides[0],padding=config.padding[0],output_padding=config.strides[0]//2),
            nn.Tanh()
        )#[3,289,289]
    def forward(self,X):
        X = self.fc1(X)
        X = self.ct1(X.view(-1,config.units[3],config.fs[0],config.fs[1]))
        X = self.ct2(X)
        X = self.ct3(X)
        return self.ct4(X)