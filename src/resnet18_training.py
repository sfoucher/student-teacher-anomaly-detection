import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from tqdm import tqdm
from argparse import ArgumentParser
from AnomalyResnet18 import AnomalyResnet18
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import load_model
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to train on (in data folder)")
    parser.add_argument('--image_size', type=int, default=256)

    # trainer arguments
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()
    return args
def freeze_layers(model, n_max= 0, learning_rate= 1e-3):
  params_to_update = []
  n_frozen, n_update= 0,0
  n_param= 0
  for l, (name,param) in enumerate(model.named_parameters()):
    n_param= n_param+1
  n_max= n_param - n_max
  for l, (name,param) in enumerate(model.named_parameters()):
    if l < int(n_max): # on controle la profondeur de la mise à jour ici
      param.requires_grad = False # freeze!
      n_frozen += 1
    else:
      n_update += 1
      param.requires_grad= True
      params_to_update.append({
                      "params": param,
                      "lr": learning_rate,
                  })
    if param.requires_grad == True:
        print("\t",name)
  print(f'frozen: {n_frozen} updated: {n_update}')
  return params_to_update



def train(args):
    # Choosing device 
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f'Device used: {device}')
    if wandb.run is not None:
      wandb.finish()
    wandb.init(project="student-teacher-anomaly-detection")
    # Resnet pretrained network for knowledge distillation
    resnet18 = AnomalyResnet18()
    resnet18.to(device)

    # Loading saved model
    model_name = f'../model/{args.dataset}/resnet18.pt'
    #load_model(resnet18, model_name)
  
    params_to_update= freeze_layers(resnet18, n_max= 17+15, learning_rate= args.learning_rate)
    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, #resnet18.parameters(), 
                          lr=args.learning_rate, 
                          momentum=args.momentum)

    # Load training data
    dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                             transform=transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(180),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                             type='train')
    train_set, val_set = train_test_split(dataset, train_size= 0.9)
    datasets = {}
    datasets['train'] = Subset(dataset, train_set)
    datasets['val'] = Subset(dataset, val_set)
    dataloader = DataLoader(datasets['train'].dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)
    dataloader_val = DataLoader(datasets['val'].dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)
    # training
    min_running_loss = np.inf
    for epoch in range(args.max_epochs):
        running_loss = 0.0
        running_corrects = 0
        max_running_corrects = 0

        for i, batch in tqdm(enumerate(dataloader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            # forward pass
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)

            # backward pass
            loss.backward()
            optimizer.step()

            # loss and accuracy
            running_loss += loss.item()
            max_running_corrects += len(targets)
            running_corrects += torch.sum(preds == targets.data)
            

        # print stats
        print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
        accuracy = running_corrects.double() / max_running_corrects
       
        running_loss_val = 0.0
        running_correctss_val = 0
        max_running_correctss_val = 0
        with torch.no_grad():
          for i, batch in tqdm(enumerate(dataloader_val)):
            # forward pass
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            # loss and accuracy
            running_loss_val += loss.item()
            max_running_correctss_val += len(targets)
            running_correctss_val += torch.sum(preds == targets.data)
        print(f"Val Epoch {epoch+1} \t loss: {running_loss_val}")
        accuracy_val = running_correctss_val.double() / max_running_correctss_val
        if running_loss_val < min_running_loss and epoch > 0:
            torch.save(resnet18.state_dict(), model_name)
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Accuracy: {accuracy}")
            print(f"Model saved to {model_name}.")
        min_running_loss = min(min_running_loss, running_loss_val)
        wandb.log({"trn/acc": 100. * accuracy,"trn/loss": running_loss,
        "val/acc": 100. * accuracy_val,"val/loss": running_loss_val})
        running_loss = 0.0
if __name__ == '__main__':
    args = parse_arguments()
    train(args)
