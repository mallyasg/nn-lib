from datasets.data_loader_provider import DataLoaderProvider
from trainer.trainer_map import model_trainer_by_model_name
import seaborn as sns
import matplotlib
from matplotlib.colors import to_rgb
import jax
import ml_collections
from absl import logging
import numpy as np

# Imports for plotting
import matplotlib.pyplot as plt

plt.set_cmap('cividis')


matplotlib.rcParams['lines.linewidth'] = 2.0

sns.reset_orig()


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
            ax[row][column].imshow(attn_maps[row][column],
                                   origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def train_and_evaluate(model_config: ml_collections,
                       data_config: ml_collections, workdir: str) -> None:
    """
      Execute the model training and evaluation.

      Args:
        config: Configuration containing model to run and its associated hyperparameters
        workdir: Directory where training related logs will be written to.

      Returns:
        None
    """
    model_name = model_config.model.name

    data_loader_provider = DataLoaderProvider(data_config)
    train_loader = data_loader_provider.get_train_loader()
    val_loader = data_loader_provider.get_val_loader()
    test_loader = data_loader_provider.get_test_loader()

    num_train_iters = len(train_loader) * model_config.num_epochs

    logging.info(f'Size of training data : {len(train_loader)} | '
                 f'Num of epochs to train : {model_config.num_epochs} | '
                 f'Num of iterations : {num_train_iters}')

    trainer = model_trainer_by_model_name[model_name](workdir, model_name,
                                                      next(iter(train_loader)),
                                                      num_train_iters,
                                                      model_config.learning_rate,
                                                      model_config.warmup,
                                                      model_config.weight_decay,
                                                      **model_config.model)
    trainer.train_model(train_loader, val_loader, model_config.num_epochs)

    val_accuracy = trainer.eval_model(val_loader)
    test_accuracy = trainer.eval_model(test_loader)

    print(f"Val accuracy:  {(100.0 * val_accuracy):4.2f}%")
    print(f"Test accuracy: {(100.0 * test_accuracy):4.2f}%")

    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})

    # data_input, labels = next(iter(val_loader))
    # inp_data = jax.nn.one_hot(data_input, num_classes=trainer.model.num_classes)
    # attention_maps = trainer.model_bd.get_attention_maps(inp_data)

    # plot_attention_maps(data_input, attention_maps, idx=0)
