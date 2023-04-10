import torch
import torch.nn as nn
from torch.optim import Adam, SGD


from train_utility_functions import get_trash_dataloaders, transforms1, load_checkpoint, pretrained_efficientnet_v2_s1, train

# setup device
GPU = torch.cuda.is_available()
device = torch.device('cpu')
if (GPU):
  print('Using gpu')
  device = torch.device('cuda')

# get data
train_loader, val_loader, test_loader = get_trash_dataloaders(transforms1())

# get the model
#model, history = load_checkpoint("efficient-v2-s-Adam_best", pretrained_efficientnet_v2_s1(device), device)
model = pretrained_efficientnet_v2_s1(device)
optimizer = Adam(model.parameters())
model.optimizer = optimizer
model.scheduler = None
history = []

# train
criterion = nn.CrossEntropyLoss()
# model = model.to(device)
train(model, criterion, train_loader, val_loader, 100, "efficient-v2-s-Adam", device, history)