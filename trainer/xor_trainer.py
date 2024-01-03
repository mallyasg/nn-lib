from models.xor_classifier import SimpleClassifier
from datasets.xor_dataset import XORDataset
from datasets.xor_dataset import xor_data_collate
from visualizer.xor_visualizer import visualize_samples, visualize_classification
import torch.utils.data as data
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm.auto import tqdm
from flax.training import checkpoints
from absl import logging

import orbax
import orbax.checkpoint
from flax.training import orbax_utils


def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=True  # Function has additional outputs, here accuracy
                                 )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc


def train_model(state, data_loader, num_epochs=100):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
    return state


def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a*b for a, b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    logging.log(logging.INFO, f"Accuracy of the model: {100.0*acc:4.2f}%")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    # Define the model
    model = SimpleClassifier(num_hidden=8, num_outputs=1)
    # Printing the model shows its attributes
    logging.log(logging.DEBUG, model)

    rng = jax.random.PRNGKey(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (8, 2))  # Batch size 8, input size 2

    # Initialize the model
    params = model.init(init_rng, inp)
    logging.log(logging.DEBUG, params)
    logging.log(logging.DEBUG, jax.tree_util.tree_map(jnp.shape, params))

    train_dataset = XORDataset(size=2500, seed=42)
    train_data_loader = data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        collate_fn=xor_data_collate)

    # Input to the optimizer are optimizer settings like learning rate
    optimizer = optax.sgd(learning_rate=0.1)

    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)

    trained_model_state = train_model(
        model_state, train_data_loader, num_epochs=100)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {'model': trained_model_state}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        'my_checkpoints/classifier.ckpt', ckpt, save_args=save_args)

    logging.log(logging.DEBUG, f"Model State is : {ckpt}")

    loaded_ckpt = orbax_checkpointer.restore(
        'my_checkpoints/classifier.ckpt')

    loaded_model_state = loaded_ckpt['model']

    logging.log(logging.DEBUG, f"Loaded Model State is : {loaded_ckpt}")

    test_dataset = XORDataset(size=500, seed=123)
    # drop_last -> Don't drop the last batch although it is smaller than 128
    test_data_loader = data.DataLoader(test_dataset,
                                       batch_size=128,
                                       shuffle=False,
                                       drop_last=False,
                                       collate_fn=xor_data_collate)

    eval_model(trained_model_state, test_data_loader)

    trained_model = model.bind(trained_model_state.params)

    visualize_classification(
        trained_model, test_dataset.data, test_dataset.label)
