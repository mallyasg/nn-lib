import os
import numpy as np
from tqdm import tqdm

import jax

from flax.training import train_state, orbax_utils
from torch.utils.tensorboard import SummaryWriter

import optax

import orbax
from orbax import checkpoint

from models.transformer.transformer_predictor import TransformerPredictor


class Trainer:

  def __init__(self,
               checkpoint_dir,
               model_name,
               example_batch,
               max_iters,
               lr=1e-3,
               warmup=100,
               seed=42,
               **model_kwargs) -> None:
    super().__init__()
    self.checkpoint_dir = checkpoint_dir
    self.model_name = model_name
    self.max_iters = max_iters
    self.lr = lr
    self.warmup = warmup
    self.seed = seed

    self.model = TransformerPredictor(**model_kwargs)

    # Setup logging dir
    self.log_dir = os.path.join(self.checkpoint_dir, self.model_name, 'log')
    self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name,
                                       'checkpoint/orbax/managed')
    self.logger = SummaryWriter(log_dir=self.log_dir)
    self.setup_checkpoint_dir()

    # Setup functions necessary for training
    self.create_trainer_functions()

    # Initialize the model with ``example_batch``
    self.initialize_model(example_batch)

  def setup_checkpoint_dir(self):
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5,
                                                        create=True)
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    self.checkpoint_manager = checkpoint.CheckpointManager(
        self.checkpoint_dir, orbax_checkpointer, options)

  def initialize_model(self, example_batch) -> None:
    self.rng = jax.random.PRNGKey(self.seed)
    self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)

    example_input = self.batch_to_input(example_batch)

    # Provide two rngs for initializing weights and dropout layer.
    params = self.model.init(
        {
            'params': init_rng,  # rng for initializing parameters
            'dropout': dropout_init_rng  # rng for initializing dropout
        },
        example_input,
        is_training=True)['params']
    # Initialize the learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                     peak_value=self.lr,
                                                     warmup_steps=self.warmup,
                                                     decay_steps=self.max_iters,
                                                     end_value=0.0)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adam(lr_schedule))
    # Initialize training state
    self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                               params=params,
                                               tx=optimizer)

  def create_trainer_functions(self) -> None:
    """
        Create necessary trainer functions -- loss calculation, train and eval steps.
        """
    calculate_loss = self.get_loss_function()

    # Function defining train step
    def train_step(state, rng, batch):

      def loss_fn(params):
        return calculate_loss(params, rng, batch, is_training=True)

      ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      loss, acc, rng = ret[0], *ret[1]
      state = state.apply_gradients(grads=grads)
      return state, rng, loss, acc

    # Function defining eval step
    def eval_step(state, rng, batch):
      _, (acc, rng) = calculate_loss(state.params,
                                     rng,
                                     batch,
                                     is_training=False)

      return acc, rng

    # Jitted train and eval steps
    self.train_step = jax.jit(train_step)
    self.eval_step = jax.jit(eval_step)

  def get_loss_function(self):
    raise NotImplementedError

  def batch_to_input(self, example_batch):
    raise NotImplementedError

  def train_model(self, train_loader, val_loader, num_epochs=500):
    best_accuracy = 0.0
    for epoch_idx in tqdm(range(1, num_epochs + 1)):
      self.train_epoch(train_loader=train_loader, epoch=epoch_idx)
      if epoch_idx % 5 == 0:
        eval_accuracy = self.eval_model(val_loader)
        self.logger.add_scalar('val/accuracy',
                               eval_accuracy,
                               global_step=epoch_idx)
        if eval_accuracy >= best_accuracy:
          best_accuracy = eval_accuracy
          self.save_model(step=epoch_idx)
        self.logger.flush()

  def train_epoch(self, train_loader, epoch):
    accuracies, losses = [], []
    for batch in tqdm(train_loader, desc='Training', leave=False):
      self.state, self.rng, loss, accuracy = self.train_step(
          self.state, self.rng, batch)
      losses.append(loss)
      accuracies.append(accuracy)

    avg_loss = np.stack(jax.device_get(losses)).mean()
    avg_accuracy = np.stack(jax.device_get(accuracies)).mean()

    self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
    self.logger.add_scalar('train/accuracy', avg_accuracy, global_step=epoch)

  def eval_model(self, data_loader):
    correct_class, count = 0, 0
    for batch in data_loader:
      accuracy, self.rng = self.eval_step(self.state, self.rng, batch)
      correct_class += accuracy * batch[0].shape[0]
      count += batch[0].shape[0]
    eval_accuracy = (correct_class / count).item()
    return eval_accuracy

  def save_model(self, step=0):
    ckpt = {'model': self.state}
    save_args = orbax_utils.save_args_from_target(ckpt)
    self.checkpoint_manager.save(step,
                                 ckpt,
                                 save_kwargs={'save_args': save_args})

  def load_model(self, pretrained=False):
    if not pretrained:
      step = self.checkpoint_manager.latest_step()  # latest step
      state = self.checkpoint_manager.restore(step)
    else:
      orbax_checkpointer = checkpoint.PyTreeCheckpointer()
      state = orbax_checkpointer.restore(
          os.path.join(self.checkpoint_dir, f'{self.model_name}.ckpt'))
    self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                               params=state.params,
                                               tx=self.state.tx)

  def checkpoint_exists(self):
    return len(os.listdir(self.checkpoint_dir)) > 0
