# Constants for easy tuning
IMG_SIZE = 32  # Input image size (assumed square)
PATCH_SIZE = 4  # Size of each patch extracted from the image
EMBED_DIM = 256  # Embedding dimension for patch embeddings and attention layers
NUM_HEADS = 8  # Number of attention heads in multi-head self-attention
DEPTH = 8  # Number of transformer encoder blocks
DROPOUT = 0.1  # Dropout rate used in the model
BATCH_SIZE = 128  # Batch size for training
LEARNING_RATE = 0.001  # Initial learning rate for optimizer
WEIGHT_DECAY = 1e-4  # Weight decay (L2 regularization) for optimizer
EPOCHS = 100  # Total number of training epochs
MAX_LR = 0.001  # Maximum learning rate for OneCycleLR scheduler
LABEL_SMOOTHING = 0.1  # Label smoothing factor for cross-entropy loss
NUM_CLASSES = 10  # Number of target classes (SVHN dataset has 10 classes)
CHECKPOINT_DIR = './checkpoints'  # Directory to store checkpoints
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait before triggering early stopping
MAX_CHECKPOINTS = 5  # Maximum number of checkpoints to keep