import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from torch.nn.parallel import DataParallel
from config import *

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_dataloaders():
    """
    Prepare and return the training and testing data loaders with data augmentation for training.

    Returns:
    train_loader (DataLoader): DataLoader for training dataset.
    test_loader (DataLoader): DataLoader for test dataset.
    """
    use_cuda = torch.cuda.is_available()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=use_cuda)

    return train_loader, test_loader


def get_latest_checkpoint():
    """
    Retrieve the path to the latest checkpoint file.

    Returns:
    str: Path to the latest checkpoint file, or None if no checkpoint exists.
    """
    checkpoints = glob.glob(f'{CHECKPOINT_DIR}/checkpoint_epoch_*.pth')
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)


def save_checkpoint(model, optimizer, epoch, best_accuracy):
    """
    Save model checkpoint and manage the number of stored checkpoints.

    Parameters:
    model (torch.nn.Module): Model to be saved.
    optimizer (torch.optim.Optimizer): Optimizer state to be saved.
    epoch (int): Current epoch number.
    best_accuracy (float): Best recorded accuracy so far.
    """
    model_to_save = model.module if isinstance(model, DataParallel) else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    torch.save(checkpoint, f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth')

    all_checkpoints = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint_epoch_*.pth'), key=os.path.getctime)
    if len(all_checkpoints) > MAX_CHECKPOINTS:
        os.remove(all_checkpoints[0])

