import torch
from torchvision import transforms, datasets, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer


######################################CHECKPOINT###########################################
def save_checkpoint(model, history:list, model_name:str, time_elapsed=0):
  """
  This function saves checkpoint of the model.

  Args:
  ====
  model: a Neural Network model that contains `epochs` field. an `optimizer`, and a `scheduler`
  history: a list of tuples (train_loss, val_loss, train_acc, val_acc)
  model name: the name of the model, used as file name when saving the checkpoint

  Return
  ====
  None 
  """
  checkpoint = {}

  checkpoint['classifier'] = model.classifier
  checkpoint['optimizer'] = model.optimizer
  checkpoint['scheduler'] = model.scheduler
  checkpoint['epochs'] = model.epochs
  try:
    model.time_to_train = model.time_to_train + time_elapsed
  except: 
    model.time_to_train = time_elapsed

  checkpoint['time_to_train'] = model.time_to_train 
  checkpoint['state_dict'] = model.state_dict()
  checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
  if model.scheduler:
    checkpoint['scheduler_state_dict'] = model.scheduler.state_dict()

  checkpoint['history'] = history
  
  path = f'./{model_name}.pt'
  torch.save(checkpoint, path)


def load_checkpoint(model_name:str, pretrained_model_with_cnnweights, device):
  '''
  This function loads a checkpoint

  Args
  ====
  model_name: the name of the file

  Returns
  =======
  model: the model attached with its `epochs` field, `optimizer`, and `scheduler`
  history: a list of tuples (train_loss, val_lost, train_acc, val_acc)
  '''
  path = f'./{model_name}.pt'
  checkpoint = torch.load(path, map_location=device)

  model = pretrained_model_with_cnnweights
  for param in model.parameters():
    param.requires_grad = False

  classifier = checkpoint['classifier']
  model.classifier = classifier
  model.load_state_dict(checkpoint['state_dict'])

  model.optimizer = checkpoint['optimizer']
  model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  model.scheduler = checkpoint['scheduler']
  if model.scheduler:
    model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  model.epochs = checkpoint['epochs']
  model.time_to_train = checkpoint['time_to_train']

  history = checkpoint['history']

  return model, history

####################################TRANSFORMS###################################
IMAGENET_MEANS = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def transforms1():
  """
  This function returns a torchvision.transforms
  """
  my_transforms = {
      'train': transforms.Compose([
          transforms.RandAugment(), 
          transforms.ToTensor(),
          transforms.Normalize(IMAGENET_MEANS, IMAGENET_STD)
      ]),
      'val': transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(IMAGENET_MEANS, IMAGENET_STD)
      ]),
      'test': transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(IMAGENET_MEANS, IMAGENET_STD)
      ])
  }

  return my_transforms

#############################################DATA LOADERS###########################################
TRAIN_FOLDER = './train'
VAL_FOLDER = './val'
TEST_FOLDER = './test'
def get_trash_dataloaders(transforms: dict, batch_size=128):
  """
  This functions return a train DataLoader, a test DataLoader, and a validation DataLoader
  Batch Size = 128
  Args:
  ====
  transforms (dict): a dictionary defining the transformations to apply on train dataset, test dataset, and val_dataset 

  Returns:
  =======
  train_loader: DataLoader
  val_loader: DataLoader
  test_loader: TestLoader
  """
  train_dataset = datasets.ImageFolder(TRAIN_FOLDER, transforms['train'])
  test_dataset = datasets.ImageFolder(TEST_FOLDER, transforms['val'])
  val_dataset = datasets.ImageFolder(VAL_FOLDER, transforms['test'])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, val_loader, test_loader

####################################TRAIN FUNCTION########################################
def train(model, criterion, train_loader, val_loader, epochs, model_name, device, history=[], max_epochs_stop=3, print_every=2, save_every=4):
  """
  This function trains a neural network. Using early stopping as overfitting occurs.

  Args:
  =====
  model: a model to train. It must be attached with a `scheduler` and an `optimizer`. Calling `model(x)` produces yhat
  criterion: a loss function
  train_loader: a DataLoader for training
  val_loader: a DataLoader for validation
  epochs: total training epochs
  model_name: used as filename when saving checkpoint
  save_every: save model after `save_every` number of epochs
  print_every: print the train/val loss  and train/val accuracy after every `print_every` number of epochs

  Return:
  ======
  None
  """
  model = model.to(device)
  start = timer()

  smallest_val_loss = np.inf
  no_improved_epochs = 0
  best_val_acc = 0 

  try:
    print(f"Model has been trained for {model.epochs} epochs.")
    start_epochs = model.epochs+1
  except:
    print("Start training from scratch.")
    start_epochs = 0
    model.epochs = 0

  for e in range(start_epochs, epochs):
    model.train()

    train_running_loss = 0
    train_running_corrects = 0
    val_running_loss = 0
    val_running_corrects = 0
    train_size = 0
    val_size = 0

    for i, (data, target) in enumerate(train_loader):
      data = data.to(device)
      target = target.to(device)

      # train
      model.optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      model.optimizer.step()

      # accumulate train loss and train acc
      _, preds = output.max(1)
      correct_preds = (preds==target).sum()
      train_running_corrects += correct_preds
      train_running_loss += loss.item()*data.size(0)
      train_size += data.size(0)
      #print(f'Epoch %d: %.2f%% completes.'%(e, (i+1)*100.0/len(train_loader)))

    else:
      model.epochs += 1

      model.eval()
      with torch.no_grad():
        for data, target in val_loader:
          data = data.to(device)
          target = target.to(device)

          # eval
          output = model(data)
          loss = criterion(output, target)

          _, preds = output.max(1)
          correct_preds = (preds==target).sum()
          val_running_corrects += correct_preds
          val_running_loss += loss.item()*data.size(0)
          val_size +=data.size(0)
        
      
    # save history for one epoch
    train_acc = train_running_corrects/train_size
    train_loss = train_running_loss/train_size
    val_acc = val_running_corrects/val_size
    val_loss = val_running_loss/val_size
    history.append((train_acc, train_loss, val_acc, val_loss))

    if (e%print_every==0):
      print(f"FINISH EPOCH {e}, training loss={train_loss:.2f}, validation loss={val_loss:.2f}")
      print(f"\t Training accuracy={train_acc*100.0:.2f}%, validation accuracy={val_acc*100.0:.2f}%")


    # save model every n epochs
    if ((e+1)%save_every==0 or e==e-1):
      print(f"Saving {model_name}.pt ...")

      time_elapsed = timer()-start
      save_checkpoint(model, history, model_name, time_elapsed)
      start = timer()

    # save the best model
    if (val_loss<smallest_val_loss):
      print(f"Saving {model_name}_best.pt ...")
      save_checkpoint(model, history, model_name+'_best')

      no_improved_epochs = 0
      smallest_val_loss = val_loss
      best_val_acc = val_acc
    else:
      no_improved_epochs += 1
    
    # early stopping
    if (no_improved_epochs==max_epochs_stop):
        print(f"FINISH TRAINING, lowest val loss is {smallest_val_loss:.2f}")
        print(f"\t Best accuracy is {best_val_acc:.2f}")
        return

  print(f"FINISH TRAINING, lowest val loss is {smallest_val_loss:.2f}")
  print(f"\t Best accuracy is {best_val_acc:.2f}") 

####################PLOT FUNCTIONS ####################################3
def plot_train_val_loss(history: pd.DataFrame, model_name, ylim: tuple, xlim: tuple):
  '''
  This function plots the train loss and validation loss

  Args:
  =====
  history (DataFrame): has two columns: `train loss` and `val loss`
  model_name: is the name of the model
  ylim (tuple): y range to zoom in
  '''
  plt.plot(range(1, len(history)+1), history['train loss'], label='train loss')
  plt.plot(range(1, len(history)+1), history['val loss'], label='validation loss')
  plt.legend()

  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  plt.title(f'Training Loss and Validation Loss of {model_name}')

  if (ylim):
    plt.ylim(ylim)

  if (xlim):
    plt.xlim(xlim)

  plt.savefig(f'loss_{model_name}.png')

  plt.show()

def plot_train_val_acc(history: pd.DataFrame, model_name, ylim: tuple, xlim: tuple):
  '''
  This functions plots the train accuracy and validation accuracy

  Args:
  =====
  history (DataFrame): has two columns: `train acc` and `val acc`
  model_name: is the name of the model
  ylim (tuple): y range to zoom in
  '''

  plt.plot(range(1, len(history)+1), history['train acc'], label='train accuracy')
  plt.plot(range(1, len(history)+1), history['val acc'], label='validation accuracy')
  plt.legend()


  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')

  plt.title(f'Training Accuracy and Validation Accuracy of {model_name}')

  if (ylim):
    plt.ylim(ylim)

  if (xlim):
    plt.xlim(xlim)

  plt.savefig(f'acc_{model_name}.png')

  plt.show()




############################## CREATE MODEL ##################################33
def pretrained_efficientnet_v2_s1(device, summarize=False):
  """
  This function returns a pretrained efficientnet_v2_s with a custom classifier
  """
  # load model
  efficientnet_v2_s = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

  # freeze parameters
  for param in efficientnet_v2_s.parameters():
    param.requires_grad = False
  
  # add custom classifier
  custom_classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=7),
    nn.Softmax(1)
  )
  efficientnet_v2_s.classifier = custom_classifier

  # to device
  efficientnet_v2_s = efficientnet_v2_s.to(device)

#   if summarize:
#     if GPU:
#       summary(efficientnet_v2_s, (3,224,224), 128, "cuda")
#     else:
#       summary(efficientnet_v2_s, (3,224,224), 128, "cpu")

  return efficientnet_v2_s