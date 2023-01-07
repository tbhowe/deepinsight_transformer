
from di_dataset import DeepInsightDataset
from model import DeepInsightVitModel
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from transformers import ViTImageProcessor
import torch


# TODO - set up transform - take PIL image, return 
# ImageProcessor output and labels in correct input format  - dict?

# TODO - init dataloaders + do train/val/test split

# TODO - init optimiser for FC layer - remember two regression heads!

# TODO - define train loop

# TODO  - define eval function

