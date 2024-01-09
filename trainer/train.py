import jax
import ml_collections
from absl import logging
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

plt.set_cmap('cividis')

from matplotlib.colors import to_rgb
import matplotlib

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns

sns.reset_orig()

from trainer.trainer_map import model_trainer_by_model_name
from datasets.dataset_map import dataset_by_model_name


def plot_attention_maps(input_data, attn_maps, idx=0):
  if input_data is not None:
    input_data = jax.device_get(input_data[idx])
  else:
    input_data = np.arange(attn_maps[0][idx].shape[-1])
  attn_maps = [jax.device_get(m[idx]) for m in attn_maps]

  num_heads = attn_maps[0].shape[0]
  num_layers = len(attn_maps)
  seq_len = input_data.shape[0]
  fig_size = 4 if num_heads == 1 else 3
  fig, ax = plt.subplots(num_layers,
                         num_heads,
                         figsize=(num_heads * fig_size, num_layers * fig_size))
  if num_layers == 1:
    ax = [ax]
  if num_heads == 1:
    ax = [[a] for a in ax]
  for row in range(num_layers):
    for column in range(num_heads):
      ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
      ax[row][column].set_xticks(list(range(seq_len)))
      ax[row][column].set_xticklabels(input_data.tolist())
      ax[row][column].set_yticks(list(range(seq_len)))
      ax[row][column].set_yticklabels(input_data.tolist())
      ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
  fig.subplots_adjust(hspace=0.5)
  plt.show()


def train_and_evaluate(config: ml_collections, workdir: str) -> None:
  """
    Execute the model training and evaluation.

    Args:
      config: Configuration containing model to run and its associated hyperparameters
      workdir: Directory where training related logs will be written to.

    Returns:
      None
  """
  model_name = config.model.name

  train_loader = dataset_by_model_name[model_name]['train']
  val_loader = dataset_by_model_name[model_name]['val']
  test_loader = dataset_by_model_name[model_name]['test']

  num_train_iters = len(train_loader) * config.num_epochs

  trainer = model_trainer_by_model_name[model_name](workdir, model_name,
                                                    next(iter(train_loader)),
                                                    num_train_iters,
                                                    config.learning_rate,
                                                    config.warmup,
                                                    **config.model)
  trainer.train_model(train_loader, val_loader, config.num_epochs)

  val_accuracy = trainer.eval_model(val_loader)
  test_accuracy = trainer.eval_model(test_loader)

  print(f"Val accuracy:  {(100.0 * val_accuracy):4.2f}%")
  print(f"Test accuracy: {(100.0 * test_accuracy):4.2f}%")

  trainer.model_bd = trainer.model.bind({'params': trainer.state.params})

  data_input, labels = next(iter(val_loader))
  inp_data = jax.nn.one_hot(data_input, num_classes=trainer.model.num_classes)
  attention_maps = trainer.model_bd.get_attention_maps(inp_data)

  plot_attention_maps(data_input, attention_maps, idx=0)
