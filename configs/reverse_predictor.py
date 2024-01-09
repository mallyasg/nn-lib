import ml_collections


def get_config() -> ml_collections.ConfigDict:
  "Get the config for reverse predictor model"
  config = ml_collections.ConfigDict()

  # Model arguments
  config.model = ml_collections.ConfigDict()
  config.model.name = "ReversePredictor"
  config.model.d_model = 32
  config.model.num_classes = 10
  config.model.num_heads = 1
  config.model.num_encoder_layers = 1
  config.model.ffn_dim_scale = 2
  config.model.dropout_prob = 0.0
  config.model.dropout_prob_for_input = 0.0

  # Hyper params
  config.warmup = 50
  config.learning_rate = 5e-4
  config.momentum = 0.9

  config.num_epochs = 10
  config.log_every_steps = 100

  # If num_train_steps == -1 then the number of training steps is calculated
  # from `num_epochs` using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  return config
