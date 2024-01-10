import math
import jax
from flax import linen as nn
from flax.linen.module import compact
from flax.linen.normalization import LayerNorm
from typing import Any, Optional
from models.transformer.attention import MultiHeadAttention
import numpy as np


class EncoderBlock(nn.Module):
    """
    Encoder module of the transformer network as described in the paper 
    Vaswani et. al., 2017 -- https://arxiv.org/abs/1706.03762
    """
    input_dim: int  # Input dimension
    num_heads: int  # Number of heads in multi-head attention
    ffn_dim: int  # Dimension of feed forward network
    dropout_prob: float  # Probability for dropout during training

    def setup(self) -> None:
        self.self_attention = MultiHeadAttention(
            num_heads=self.num_heads, out_dim=self.input_dim,
            qkv_dim=self.input_dim, normalize_qk=False)

        self.dropout_layer = nn.Dropout(rate=self.dropout_prob)

        self.feed_forward_1 = nn.Dense(
            name='ffn_layer_1', features=self.ffn_dim)

        self.feed_forward_2 = nn.Dense(
            name='ffn_layer_2', features=self.input_dim)

        self.attention_layer_norm = LayerNorm(name='add_and_norm_attention')

        self.feed_forward_layer_norm = LayerNorm(name='add_and_norm_ffn')

    @compact
    def __call__(self, x, mask=None, is_training=True):
        self_attn_out, self_attn_weights = self.self_attention(x, mask=mask)

        deterministic = not is_training

        x = self.attention_layer_norm(
            x + self.dropout_layer(self_attn_out, deterministic=deterministic))

        return self.feed_forward_layer_norm(
            x + self.dropout_layer(
                self.feed_forward_2(
                    self.dropout_layer(
                        self.feed_forward_1(x),
                        deterministic=deterministic)),
                deterministic=deterministic))


class Encoder(nn.Module):
    """
    Encoder part of transformer.
    """
    num_encoder_layers: int  # Number of encoder layers to stack
    input_dim: int  # Input dimension
    num_heads: int  # Number of heads in multi-head attention
    ffn_dim: int  # Dimension of feed forward network
    dropout_prob: float  # Probability for dropout during training

    def setup(self) -> None:
        self.encoder_blocks = [
            EncoderBlock(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout_prob=self.dropout_prob)
            for _ in range(self.num_encoder_layers)]

    def __call__(self, x, mask=None, is_training=True):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask=mask, is_training=is_training)
        return x

    def get_attention_maps(self, x, mask=None, is_training=True):
        """
        Generates the attention map for visualization
        """
        attention_maps = []
        for encoder_block in self.encoder_blocks:
            _, attn_map = encoder_block.self_attention(x, mask=mask)
            attention_maps.append(attn_map)
            x = encoder_block(x, mask=mask, is_training=is_training)
        return attention_maps


class PositionalEncoder(nn.Module):
    """
    ``PositionalEncoder`` encodes the position information to the input feature 
    vector.
    """
    d_model: int  # Hidden dimensionality of the model.
    max_len: int = 5000  # Maximum of length of the sequence to expect.

    def setup(self) -> None:
        positional_encoding = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) *
            (-math.log(10000.0) / self.d_model))
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        positional_encoding = positional_encoding[None]
        self.positional_encoding = jax.device_put(positional_encoding)

    def __call__(self, x):
        return x + self.positional_encoding[:, :x.shape[1]]
