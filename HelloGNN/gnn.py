# Time: 2021/8/2 20:09
# Software: PyCharm
# Description: gnn
from torch import nn
from torchvision.transforms import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os, random, glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from functools import partial
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
# torch.cuda.manual_seed(random_state)
# torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
use_gpu = torch.cuda.is_available()


import torch
from torch_geometric.data import Data

#边，shape = [2,num_edge]
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

#点，shape = [num_nodes, num_node_features]
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

