import torch
from pathlib import Path
import numpy as np
import os, time, sys, copy, gc
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam, lr_scheduler

from utils import generatePlots, getModelName, epoch_time
from services import EarlyStopper

def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        if len(batch) == 2:
            imgs, labels_y = batch
        else:
            imgs, labels_y, _ = batch  # Ignore extra data if present

        imgs = imgs.to(device)
        labels_y = labels_y.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        _, pred = torch.max(output.data, 1)
        loss = criterion(output, labels_y)
        loss.backward()
        total_loss += loss.item() * imgs.size(0)
        correct += torch.sum(pred == labels_y.data)
        total += labels_y.size(0)
        optimizer.step()
        
        del imgs, labels_y, output
        gc.collect()
        torch.cuda.empty_cache()
 
    return correct / total, total_loss / len(train_loader)

def evaluate(model, criterion, val_loader, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 2:
                imgs, labels_y = batch
            else:
                imgs, labels_y, _ = batch  # Ignore extra data if present

            imgs = imgs.to(device)
            labels_y = labels_y.to(device)
            output = model(imgs)
            _, pred = torch.max(output.data, 1)
            loss = criterion(output, labels_y)
            correct += torch.sum(pred == labels_y.data)
            epoch_loss += loss.item() * imgs.size(0)
            total += labels_y.size(0)
            
            del imgs, labels_y, output
            gc.collect()
            torch.cuda.empty_cache()
 
    return correct / total, epoch_loss / len(val_loader)

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parent = '/content/banglalekha_dataset/BanglaLekha_Dataset/'
    train_folder = os.path.join(parent, 'train')
    val_folder = os.path.join(parent, 'test')
    
    if not os.path.exists(train_folder) or not os.path.exists(val_folder):
        sys.exit('Data folder not available')
    
    model_name = getModelName(config)
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize all images to 224x224 for ResNet
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization for ResNet
    ])
    
    train_dataset = ImageFolder(root=train_folder, transform=transform)
    val_dataset = ImageFolder(root=val_folder, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    num_classes = len(train_dataset.classes)
    model = resnet34(pretrained=False)
    model.fc = Linear(512, num_classes, bias=True)
    model.to(device)
    
    criterion = CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0004)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    earlystopper = EarlyStopper(patience=config.patience)
    
    best_valid_loss = np.inf
    c = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch+1:02}\tlearning rate: {scheduler.get_last_lr()}\n")
        train_acc, train_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        val_acc, val_loss = evaluate(model, criterion, val_loader, device)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            checkpoints_dir = os.path.join(parent, 'Checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)  # Ensure the directory exists
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"{model_name}.pth"))
            print(f"Model saved with Validation loss: {val_loss:.4f}\n")
            c = 0
        else:
            c += 1
        
        if c == 5:
            scheduler.step()
            c = 0
        
        if earlystopper.early_stop(val_loss):
            print("Model is not improving. Quitting ...")
            break
        
        torch.cuda.empty_cache()
    
    print(f"For inference, use model: {model_name}\n")
