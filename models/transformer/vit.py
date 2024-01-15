from typing import Any
from flax import linen as nn
import torch
import jax.numpy as jnp


class AttentionBlock(nn.Module):
  embed_dim: int  # Dimension of the input and attention feature vectors
  hidden_dim: int  # Dimension of the hidden units in the feed-forward network
  num_heads: int  # Number of heads for multi-headed attention
  dropout_prob: float  # Probability for dropout during training

  def setup(self) -> None:
    self.attention = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.embed_dim,
        out_features=self.embed_dim)
    self.linear = [
        nn.Dense(self.hidden_dim), nn.gelu,
        nn.Dropout(self.dropout_prob),
        nn.Dense(self.embed_dim)
    ]
    self.layer_norm_1 = nn.LayerNorm()
    self.layer_norm_2 = nn.LayerNorm()
    self.dropout = nn.Dropout(self.dropout_prob)

  def __call__(self, x, is_training=True):
    deterministic = not is_training
    layer_norm_out = self.layer_norm_1(x)
    x = x + self.dropout(self.attention(inputs_q=layer_norm_out,
                                        inputs_kv=layer_norm_out),
                         deterministic=deterministic)

    feed_forwad_net_out = self.layer_norm_2(x)
    for layer in self.linear:
      feed_forwad_net_out = layer(feed_forwad_net_out) if not isinstance(
          layer, nn.Dropout) else layer(feed_forwad_net_out,
                                        deterministic=deterministic)
    x = x + self.dropout(feed_forwad_net_out, deterministic=deterministic)
    return x


def img_to_patch(x: torch.Tensor,
                 patch_size: int,
                 flatten_channels: bool = True):
  """
  Converts the image to patches of size ``patch_size`` x ``patch_size``.

  Args:
    x : batch of images with shape (batch_size, height, width, channels)
    patch_size : Number of pixels per in each dimension
    flatten_channels: If true, the image patch will be in the form of feature 
                      vector instead of 2D grid 
  """
  B, H, W, C = x.shape
  x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size,
                C)  # (B, H', p_H, W', p_W, C)
  x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H', W', p_H, p_W, C)
  x = x.reshape(B, -1, *x.shape[3:])  # (B, H' * W', p_H, p_W, C)

  if flatten_channels:
    x = x.reshape(B, x.shape[1], -1)  # (B, H' * W', p_H * p_W * C)

  return x


class ViT(nn.Module):
  """
  Defines the Vision Transformer model described in Alexey Dosovitskiy et al. 
  https://openreview.net/pdf?id=YicbFdNTTy.
  """

  embed_dim: int  # Dimension of input and attention feature vectors
  hidden_dim: int  # Dimension of the hidden units in the feed-forward network
  num_heads: int  # Number of heads in multi-head attention
  num_channels: int  # Number of channels in the input image
  num_layers: int  # Number of encoder/attention blocks to use in transformer
  num_classes: int  # Number of classes for classification
  patch_size: int  # Number of pixels that the patches have per dimension
  num_patches: int  # Maximum number of patches an image can have
  dropout_prob: float = 0.0  # Probability for dropout during training

  def setup(self) -> None:
    """
    Setup the layers in ViT
    """
    self.input_layer = nn.Dense(features=self.embed_dim)
    self.transformer = [
        AttentionBlock(self.embed_dim, self.hidden_dim, self.num_heads,
                       self.dropout_prob) for _ in range(self.num_layers)
    ]
    self.mlp_head = nn.Sequential(
        [nn.LayerNorm(), nn.Dense(features=self.num_classes)])

    self.dropout = nn.Dropout(self.dropout_prob)

    self.cls_token = self.param('cls_token', nn.initializers.normal(stddev=1.0),
                                (1, 1, self.embed_dim))

    self.positional_embedding = self.param(
        'positional_embedding', nn.initializers.normal(stddev=1.0),
        (1, 1 + self.num_patches, self.embed_dim))

  def __call__(self, x, is_training=True):
    x = img_to_patch(x, self.patch_size)
    B, T, _ = x.shape
    x = self.input_layer(x)

    # Add CLS token and positional encoding
    cls_token = self.cls_token.repeat(B, axis=0)
    x = jnp.concatenate([cls_token, x], axis=1)

    x = x + self.positional_embedding[:, :T + 1]

    x = self.dropout(x, deterministic=not is_training)
    for attention_block in self.transformer:
      x = attention_block(x, is_training)

    cls = x[:, 0]
    out = self.mlp_head(cls)

    return out
