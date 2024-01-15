import jax
import optax
from trainer.trainer import Trainer


class ViTTrainer(Trainer):

  def batch_to_input(self, batch):
    return batch[0]

  def get_loss_function(self):

    def calculate_loss(params, rng, batch, is_training):
      input_data, labels = batch
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

  def init_optimizer(self):
    # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=self.lr,
        boundaries_and_scales={
            int(self.max_iters * 0.6): 0.1,
            int(self.max_iters * 0.85): 0.1
        })

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
        optax.adamw(lr_schedule, weight_decay=self.weight_decay))

    return optimizer
