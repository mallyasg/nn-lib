import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.module import compact
from flax.linen.normalization import LayerNorm
from typing import Any, Optional
from absl import logging

Array = Any


def ScaledDotProductAttention(
        query, key, value, mask=None, return_attention_weights=True):
    """
    Computes the scaled dot product attention.

    ``ScaledDotProductAttention`` takes query, key and value vectors as the 
    first three arguments. Optionally a mask can also be provided as input. Mask
    is useful when vectors of different lengths are batched together.
    """
    d_k = query.shape[-1]
    # attention = softmax(QK^T / √d_k)V
    attn = jnp.matmul(query, jnp.swapaxes(key, -2, -1))
    attn = attn / math.sqrt(d_k)

    if mask is not None:
        attn = jnp.where(mask == 0, -9e15, attn)

    attn = nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, value)

    if return_attention_weights:
        return out, attn
    else:
        return out


def expand_mask(mask):
    """
    Helper function to support different mask shapes.
    Output shape supports (batch_size, number of heads, seq length, seq length)
    If 2D: broadcasted over batch size and number of heads
    If 3D: broadcasted over number of heads
    If 4D: leave as is
    """
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiHeadAttention(nn.Module):
    num_heads: int  # Number of heads in the multi-headed attention
    out_dim: Optional[int] = None  # Dimension of the output
    # Dimension of the input query, key and value feature vectors
    qkv_dim: Optional[int] = None
    normalize_qk: Optional[bool] = False

    @compact
    def __call__(
            self, input_q: Array, input_k: Optional[Array] = None,
            input_v: Optional[Array] = None,
            mask: Optional[Array] = None):
        batch_size, sequence_length, input_query_dim = input_q.shape

        if mask is not None:
            mask = expand_mask(mask)

        if input_k is None:
            if input_v is not None:
                raise ValueError(
                    "``input_k`` cannot be None when ``input_v`` is not None.")
            input_k = input_q

        if input_v is None:
            input_v = input_k

        # Validate ``batch_size`` and ``sequence_length`` values of input key and value vectors
        assert (batch_size, sequence_length) == input_k.shape[:-1]
        assert (batch_size, sequence_length) == input_v.shape[:-1]

        input_key_dim = input_k.shape[-1]
        input_val_dim = input_v.shape[-1]

        qkv_dim = self.qkv_dim or input_query_dim
        out_dim = self.out_dim or input_query_dim

        assert qkv_dim % self.num_heads == 0, (
            f'Feature Dimension ({qkv_dim}) must be divisible by number of '
            f'heads ({self.num_heads})')

        head_dim = qkv_dim // self.num_heads

        # Project the input query, key and value vectors
        if input_query_dim == input_key_dim and input_query_dim == input_val_dim:
            query = nn.Dense(features=qkv_dim)(input_q).reshape(
                batch_size, sequence_length, self.num_heads, -1).transpose(0, 2, 1, 3)
            key = nn.Dense(features=qkv_dim)(input_k).reshape(
                batch_size, sequence_length, self.num_heads, -1).transpose(0, 2, 1, 3)
            value = nn.Dense(features=qkv_dim)(input_v).reshape(
                batch_size, sequence_length, self.num_heads, -1).transpose(0, 2, 1, 3)
        else:
            qkv_projection = nn.Dense(
                features=3 * qkv_dim)(jax.numpy.concatenate([input_q, input_k, input_v], axis=2))

            qkv_projection = qkv_projection.reshape(
                batch_size, sequence_length, self.num_heads, -1)

            # [batch_size, num_heads, seq_length, 3 * dim]
            qkv_projection = qkv_projection.transpose(0, 2, 1, 3)

            query, key, value = jnp.array_split(qkv_projection, 3, axis=-1)

        if self.normalize_qk:
            # As mentioned in ViT-22B (https://arxiv.org/pdf/2302.05442.pdf)
            # normalizing the query/key helps in stabilizing the training at
            # scale.
            query = LayerNorm(name='normalize_query', use_bias=False)(query)
            key = LayerNorm(name='normalize_key', use_bias=False)(key)

        # Compute scaled dot product attention using query, key and value vectors
        attention_values, attention_weights = ScaledDotProductAttention(
            query=query,
            key=key,
            value=value,
            mask=mask)

        output = nn.Dense(
            features=out_dim)(
            attention_values.reshape(batch_size, sequence_length, qkv_dim))

        return output, attention_weights
