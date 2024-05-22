import os
import sys
sys.path.append('/sources')
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import time
import random
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from config.res18_cifar10_whole import Config
from utils.dataset import InfoCIFAR10
from utils.reverse_policy import ReversePruningPolicy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_train():
    max_accuracy = 0.
    train_duration = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")
        scheduler = OneCycleLR(optimizer, max_lr=config.max_lr, total_steps=config.num_epochs * len(train_loader), pct_start=config.pct_start, last_epoch=epoch * len(train_loader) - 1)

        st = time.time()
        iter_count = 0
        for idx, (x, y, sample_idx, scaler) in progress_bar:
            x, y, scaler = x.to(device), y.to(device), scaler.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = train_criterion(logits, y)

            policy.update_scores(sample_idx, loss.detach().cpu())

            loss = torch.mean(loss*scaler)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': train_loss/(idx+1)})
            iter_count += 1
        
        epoch_time = time.time() - st
        train_duration += epoch_time
        policy.update_policy(epoch+1)
        
        # test
        test_loss, test_accuracy = _run_test()
        print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

        mlflow.log_metrics({'train_loss': train_loss / len(train_loader), 
                            'test_loss': test_loss,
                            'test_accuracy': test_accuracy,
                            'learing_rate': scheduler.get_last_lr()[0],
                            'epoch_time': epoch_time,
                            'train_duration': train_duration,
                            'threshold': policy.threshold.item(),
                            'iter_count': iter_count}, step=epoch)

        if test_accuracy > max_accuracy:
            mlflow.pytorch.log_model(model, 'best_model')
            max_accuracy = test_accuracy

def _run_test():
    model.eval()
    test_loss = 0.
    num_correct = 0
    count = 0
    
    with torch.no_grad():
        for idx, (x ,y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = test_criterion(logits, y)

            test_loss += loss.item()
            _, pred = logits.max(1)
            num_correct += pred.eq(y).sum().item()
            count += y.size(0)
        
        return test_loss / len(test_loader), num_correct / count


if __name__ == "__main__":
    seed_everything(seed=42)

    config = Config()

    # Data related
    train_tranform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*config.stats)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*config.stats)
    ])
    train_dataset = InfoCIFAR10(root='/sources/dataset/cifar10', train=True, download=True, transform=train_tranform)
    test_datset = datasets.CIFAR10(root='/sources/dataset/cifar10', train=False, download=True, transform=test_transform)

    # Policy
    policy = ReversePruningPolicy(len(train_dataset), config.num_epochs, prob=0.7)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, sampler=policy)
    test_loader = DataLoader(test_datset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # model related
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smooth, reduction='none')
    test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.max_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    # scheduler = OneCycleLR(optimizer, max_lr=config.max_lr, total_steps=config.num_epochs * len(train_loader), pct_start=config.pct_start)

    # mlflow setting
    mlflow.set_experiment('R18_CIFAR10_IB_REV')
    mlflow.start_run()
    mlflow.log_param("num_epochs", config.num_epochs)
    mlflow.log_param("bath_size", config.batch_size)
    mlflow.log_param("max_lr", config.max_lr)
    mlflow.log_param('pct_start', config.pct_start)
    mlflow.log_param('label_smooth', config.label_smooth)
    mlflow.log_param('prune_prob', policy.prob)
    mlflow.log_param('anneal', policy.anneal)

    run_train()

    mlflow.end_run()