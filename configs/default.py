import ml_collections


def get_default_config() -> ml_collections.ConfigDict:
    "Get the default values for hyperparameters"
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.1
    config.momentum = 0.9

    config.num_epochs = 100.0
    config.log_every_steps = 100

    # If num_train_steps == -1 then the number of training steps is calculated
    # from `num_epochs` using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config
