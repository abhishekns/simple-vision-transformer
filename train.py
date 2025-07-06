import torch
from vit_model import SimpleViT
from utils import *

def train_and_evaluate(parallel: bool = True):
    """
    Main function to train and evaluate the Simple Vision Transformer model on the SVHN dataset.
    Parameters:
    parallel (bool): Whether to use DataParallel for multi-GPU training.
    """
    train_loader, test_loader = get_dataloaders()
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SimpleViT(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, depth=DEPTH, dropout=DROPOUT)
    model = model.to(device)
    train(train_loader, device, model, writer, parallel)
    evaluate(model, test_loader, device)
    writer.close()


def train(train_loader: DataLoader, device: torch.device, model: SimpleViT, writer: SummaryWriter, parallel: bool = True):
    """
    Train the model and evaluate performance with restartable training, checkpointing, and early stopping.
    Parameters:
    train_loader (DataLoader): DataLoader for the training dataset.
    device (torch.device): Device to perform training on.
    model (SimpleViT): The Vision Transformer model to be trained.
    writer (SummaryWriter): TensorBoard writer for logging.
    parallel (bool): Whether to use DataParallel for multi-GPU training.
    """

    parallelized = torch.cuda.device_count() > 1 and parallel
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if parallelized:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = DataParallel(model)

    start_epoch = 0
    best_accuracy = 0.0
    early_stop_counter = 0
    total_steps = EPOCHS * len(train_loader)

    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        if parallelized:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resumed training from epoch {start_epoch}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=total_steps)

    for epoch in range(start_epoch, EPOCHS):
        print(f"Starting training from epoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            #print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            # print(f"Data shape: {data.shape}, Target shape: {target.shape}")
            output, _ = model(data)
            loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)(output, target)
            #print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            # print(f"Optimizer step completed for batch {batch_idx + 1}")
            scheduler.step()
            #print(f"Learning rate after step: {scheduler.get_last_lr()[0]}")

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            writer.add_scalar('Batch Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        accuracy = 100.0 * correct / total
        writer.add_scalar('Epoch Loss', total_loss / len(train_loader), epoch)
        writer.add_scalar('Epoch Accuracy', accuracy, epoch)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

        save_checkpoint(model, optimizer, epoch, best_accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
        return model

def evaluate(model, test_loader, device):
    """
    Evaluate the model's performance on the test dataset.

    Parameters:
    model (torch.nn.Module): Trained model to be evaluated.
    test_loader (DataLoader): DataLoader for the test dataset.
    device (torch.device): Device to perform evaluation on.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs, _ = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
