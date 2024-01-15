import jax
import optax
from trainer.trainer import Trainer
from flax.training import train_state, orbax_utils


class ReverseTrainer(Trainer):

  def batch_to_input(self, batch):
    input_data, _ = batch
    input_data = jax.nn.one_hot(input_data, num_classes=self.model.num_classes)

    return input_data

  def get_loss_function(self):

    def calculate_loss(params, rng, batch, is_training):
      input_data, labels = batch
      input_data = jax.nn.one_hot(input_data,
                                  num_classes=self.model.num_classes)

      rng, dropout_rng = jax.random.split(rng)
      logits = self.model.apply({'params': params},
                                input_data,
                                is_training=is_training,
                                rngs={'dropout': dropout_rng})
      loss = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits, labels=labels).mean()
      accuracy = (logits.argmax(axis=-1) == labels).mean()
      return loss, (accuracy, rng)

    return calculate_loss

  def init_optimizer(self, params):
    # Initialize the learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                     peak_value=self.lr,
                                                     warmup_steps=self.warmup,
                                                     decay_steps=self.max_iters,
                                                     end_value=0.0)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adam(lr_schedule))

    return optimizer
