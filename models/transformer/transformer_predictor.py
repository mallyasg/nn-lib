from flax import linen as nn
from models.transformer.transformer import PositionalEncoder, Encoder


class TransformerPredictor(nn.Module):
    """
    Encoder only transformer module for performing classification based tasks.
    """
    d_model: int  # Dimension of the model
    num_classes: int  # Number of classes in the classification task
    num_heads: int  # Number of heads in the multi-head attention
    num_encoder_layers: int  # Number of encoder layers to stack on top of each other
    dropout_prob: float  # Probability for dropout during training
    dropout_prob_for_input: float  # Probability for dropout in the input layer
    ffn_dim_scale: int  # Scale factor for scaling ``d_model`` to be used as ffn dimension

    def setup(self) -> None:
        self.input_projection = nn.Dense(features=self.d_model)
        self.input_dropout = nn.Dropout(rate=self.dropout_prob_for_input)
        self.positional_encoder = PositionalEncoder(d_model=self.d_model)
        self.transformer = Encoder(
            num_encoder_layers=self.num_encoder_layers, input_dim=self.d_model,
            ffn_dim=(self.ffn_dim_scale * self.d_model),
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob)
        self.output_net = [
            nn.Dense(self.d_model),
            nn.LayerNorm(name='output_norm'),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes)
        ]

    def __call__(self, x, mask=None, add_positional_encoding=True,
                 is_training=True):
        """
        Args:
            x: Input features of shape [batch_size, sequence_length, input_dim]
            mask: Mask to apply on the attention outputs
        """
        deterministic = not is_training
        x = self.input_dropout(x, deterministic=deterministic)
        x = self.input_projection(x)

        if add_positional_encoding:
            x = self.positional_encoder(x)

        x = self.transformer(x, mask=mask, is_training=is_training)

        for layer in self.output_net:
            if isinstance(layer, nn.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x

    def get_attention_maps(
            self, x, mask=None, add_positional_encoding=True, is_training=True):
        deterministic = not is_training
        x = self.input_dropout(x, deterministic=deterministic)
        x = self.input_projection(x)
        if add_positional_encoding:
            x = self.positional_encoder(x)
        attention_maps = self.transformer.get_attention_maps(
            x, mask=mask, is_training=is_training)
        return attention_maps
