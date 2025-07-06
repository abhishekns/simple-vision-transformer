# Simple Vision Transformer (ViT) for Image Classification

This project implements a **simple yet scalable Vision Transformer (ViT)** from scratch in PyTorch and applies it to image classification tasks such as **SVHN** (Street View House Numbers). The model includes key features like:
- Custom Vision Transformer architecture
- Data augmentation
- Learning rate scheduling
- Restartable training with checkpointing
- Multi-GPU support via `DataParallel`
- TensorBoard logging
- Early stopping
- Parameterized configuration for easy tuning

## Project Structure
    .
    ├── vit_model.py # Vision Transformer model components
    ├── train.py # Training and evaluation workflow (to be modularized)
    ├── utils.py # Data loading, checkpointing utilities (to be modularized)
    ├── checkpoints/ # Directory for model checkpoints
    └── README.md # Project documentation

## Requirements
- Python 3.8+
- PyTorch >= 1.10
- torchvision
- TensorBoard

You can install the dependencies via:
```bash
pip install torch torchvision tensorboard
```

### Key Features
- Vision Transformer (ViT): Implements patch embedding, multi-head self-attention, and transformer encoder blocks.
- Data Augmentation: Uses color jitter, affine transformations, random erasing, and more.
- Learning Rate Scheduler: OneCycleLR scheduler for fast convergence.
- Restartable Training: Automatically loads the latest checkpoint to resume training.
- TensorBoard Support: Logs batch and epoch metrics.
- Parallel Training: Supports multi-GPU via DataParallel.
- Early Stopping: Stops training when validation accuracy stops improving.

### Usage
#### Training the Model

    python train.py

The model will automatically resume from the latest checkpoint if available.

#### TensorBoard
    tensorboard --logdir=runs

Open http://localhost:6006/ in your browser to monitor training progress.

#### Configuration
  All key parameters like batch size, learning rate, patch size, embedding dimension, and dropout rates are easily configurable within the training script.

Example parameters:

    IMG_SIZE = 32
    PATCH_SIZE = 4
    EMBED_DIM = 256
    NUM_HEADS = 8
    DEPTH = 8
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 100

#### To Do
- Modularize training, data loading, and utilities into separate files.
- Add mixed precision training.
- Extend support to other datasets like CIFAR-10, CIFAR-100.

## License 
This project is released under the MIT License.

## Acknowledgements
This project is inspired by the original Vision Transformer (ViT) paper by Dosovitskiy et al.
