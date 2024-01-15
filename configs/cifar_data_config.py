import ml_collections


def get_config() -> ml_collections.ConfigDict:
  "Get the config for CIFAR-10 data"
  config = ml_collections.ConfigDict()

  # Data arguments
  config.name = "CIFAR10"
  config.path = "./data/CIFAR10/"

  # Transforms
  config.transforms = ml_collections.ConfigDict()
  config.transforms.normalize_image = True
  config.transforms.image_mean = [0.49139968, 0.48215841, 0.44653091]
  config.transforms.image_std = [0.24703223, 0.24348513, 0.26158784]

  config.transforms.perform_random_horizontal_flip = True

  # Transforms > Random Resized Crops
  config.transforms.random_resized_crop = ml_collections.ConfigDict()
  config.transforms.random_resized_crop_size = (32, 32)
  config.transforms.random_resized_crop_scale = (0.8, 1.0)
  config.transforms.random_resized_crop_ratio = (0.9, 1.1)

  return config
