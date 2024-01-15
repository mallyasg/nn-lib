import ml_collections


def get_config() -> ml_collections.ConfigDict:
  "Get the config for reverse predictor model"
  config = ml_collections.ConfigDict()

  # Model arguments
  config.model = ml_collections.ConfigDict()
  config.model.name = "ViT"
  config.model.embed_dim = 256
  config.model.hidden_dim = 512
  config.model.num_heads = 8
  config.model.num_classes = 10
  config.model.num_channels = 3
  config.model.num_layers = 6
  config.model.patch_size = 4
  config.model.num_patches = 64
  config.model.dropout_prob = 0.2

  # Hyper params
  config.warmup = 50
  config.learning_rate = 3e-4
  config.weight_decay = 0.01

  config.num_epochs = 200
  config.log_every_steps = 100

  # If num_train_steps == -1 then the number of training steps is calculated
  # from `num_epochs` using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  return config
