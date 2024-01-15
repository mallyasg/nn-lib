import ml_collections


def get_config() -> ml_collections.ConfigDict:
  "Get the config for CIFAR-10 data"
  config = ml_collections.ConfigDict()

  # Data arguments
  config.name = "ReversePredictor"
  config.data_params = ml_collections.ConfigDict()
  config.data_params.num_classes = 10
  config.data_params.seq_len = 16
  config.data_params.num_train_data_points = 50000
  config.data_params.num_val_data_points = 1000
  config.data_params.num_test_data_points = 10000

  return config
