#%%
from di_dataset import DeepInsightDataset
from model import DeepInsightVitModel
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
import torch


# TODO - init dataloaders + do train/val/test split

dataset = DeepInsightDataset()
train_set_len = round(0.8*len(dataset))
val_set_len = round(0.1*len(dataset))
test_set_len = len(dataset) - val_set_len - train_set_len
split_lengths = [train_set_len, val_set_len, test_set_len]
# split the data to get validation and test sets
train_set, val_set, test_set = random_split(dataset, split_lengths)

batch_size = 16
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# TODO - init optimiser for FC layer - remember two regression heads!


# TODO - define train loop

# TODO  - define eval function


# %%
