# vit_model.py
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention module for Vision Transformers.

    Parameters:
    embed_dim (int): Total embedding dimension.
    num_heads (int): Number of attention heads.
    dropout (float): Dropout rate to apply after attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Multi-Head Self Attention.

        Parameters:
        x (torch.Tensor): Input tensor of shape (B, N, D)

        Returns:
        Tuple of attention output and attention weights.
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        output = self.proj(attn_output)

        return output, attn_weights


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block combining Multi-Head Self Attention and Feedforward network.

    Parameters:
    embed_dim (int): Total embedding dimension.
    num_heads (int): Number of attention heads.
    mlp_ratio (float): Expansion factor for the feedforward layer.
    dropout (float): Dropout rate for both attention and MLP layers.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Transformer Encoder Block.

        Parameters:
        x (torch.Tensor): Input tensor of shape (B, N, D)

        Returns:
        Tuple of output tensor and attention weights.
        """
        attn_output, attn_weights = self.mhsa(self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class PatchEmbedding(nn.Module):
    """
    Converts image into a sequence of patch embeddings.

    Parameters:
    img_size (int): Size of input image (assumed square).
    patch_size (int): Size of each square patch.
    in_channels (int): Number of input channels.
    embed_dim (int): Embedding dimension for each patch.
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.

        Parameters:
        x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
        Embedded patch sequence tensor of shape (B, N, D)
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleViT(nn.Module):
    """
    Simple Vision Transformer (ViT) model.

    Parameters:
    img_size (int): Size of input image.
    patch_size (int): Size of each patch.
    in_channels (int): Number of input channels.
    num_classes (int): Number of output classes.
    embed_dim (int): Embedding dimension for patches.
    num_heads (int): Number of attention heads.
    depth (int): Number of Transformer Encoder Blocks.
    dropout (float): Dropout rate.
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, num_classes: int = 10, embed_dim: int = 256, num_heads: int = 8, depth: int = 8, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the Vision Transformer.

        Parameters:
        x (torch.Tensor): Input image batch of shape (B, C, H, W)

        Returns:
        Tuple of logits tensor and list of attention maps from each encoder block.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]

        attention_maps = []
        for block in self.encoder_blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)

        cls_output = x[:, 0]
        out = self.mlp_head(cls_output)
        return out, attention_maps
