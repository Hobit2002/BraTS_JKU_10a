import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock
import time, wandb
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

class ViTModel(nn.Module):
    def __init__(self, input_nc = 2, output_nc = 1, hidden_dim=256, num_heads=8, depth=6, patch_size=16):
        super(ViTModel, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Assuming cubic volumes, height == width == depth
        patch_volume = self.patch_size ** 3  # Adjust as needed
        inp_height, inp_width, inp_depth = 128, 128, 96
        self.num_patches_h = inp_height // patch_size
        self.num_patches_w = inp_width // patch_size
        self.num_patches_d = inp_depth // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w * self.num_patches_d
        
        # Patch embedding for 3D patches
        self.patch_embedding = nn.Conv3d(input_nc, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # Transformer Encoder
        self.encoder = nn.Sequential(
            *[EncoderBlock(num_heads, hidden_dim, hidden_dim * 4, dropout=0.1, attention_dropout=0.25) for _ in range(depth)]
        )
        
        # Upsampler using transposed convolution instead of linear layer
        self.upsampler = nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # Reconstruction to 3D volume
        self.reconstruction = nn.Conv3d(hidden_dim, output_nc, kernel_size=3, padding=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (B, hidden_dim, D/P, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)
        
        # Add positional embeddings
        x = x + self.positional_embedding
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Reshape back to volume form with correct block alignment
        x = x.transpose(1, 2).view(batch_size, self.hidden_dim, self.num_patches_d, self.num_patches_h, self.num_patches_w)
        
        # Apply transposed convolution for upsampling
        x = self.upsampler(x)
        
        # Final reconstruction layer
        x = self.reconstruction(x)
        x = x.transpose(4,2)

        return x


class ViTModel_Old(nn.Module):
    def __init__(self, input_nc = 2, output_nc = 1, hidden_dim=256, num_heads=8, depth=6, patch_size=16):
        super(ViTModel_Old, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.patch_size = patch_size
        
        # Assuming cubic volumes, height == width == depth
        patch_volume = self.patch_size ** 3  # Adjust as needed
        inp_height, inp_width, inp_depth = 128, 128, 96
        inp_volume = inp_height * inp_width * inp_depth
        self.num_patches_h = inp_height // patch_size
        self.num_patches_w = inp_width // patch_size
        self.num_patches_d = inp_depth // patch_size
        self.num_patches = inp_volume // patch_volume
        
        # Patch embedding for 3D patches
        self.patch_embedding = nn.Conv3d(input_nc, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # Transformer Encoder
        self.encoder = nn.Sequential(
            *[EncoderBlock(num_heads, hidden_dim, hidden_dim * 4, dropout=0.1, attention_dropout=0.25) for _ in range(depth)]
        )
        self.upsampler = nn.Linear(hidden_dim, patch_size**3, bias=False)
        
        # Reconstruction to 3D volume
        self.reconstruction = nn.ConvTranspose3d(output_nc, output_nc, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (B, hidden_dim, D/P, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)
        
        # Add positional embeddings
        x = x + self.positional_embedding
        
        # Transformer encoding
        x = self.encoder(x)
        x = self.upsampler(x)
        
        # Reshape back to volume form
        x = x.view(batch_size, self.num_patches_h, self.num_patches_w, self.num_patches_d, self.patch_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(batch_size, 1, self.num_patches_h * self.patch_size, self.num_patches_w * self.patch_size, self.num_patches_d * self.patch_size)
        x = self.reconstruction(x)
        
        return x
