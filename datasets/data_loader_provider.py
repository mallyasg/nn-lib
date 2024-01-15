import torch.utils.data as data
import ml_collections
from datasets.cifar_dataset import CIFAR10DataLoader
from datasets.reverse_dataset import ReverseDataLoader


class DataLoaderProvider:
  data_config: ml_collections

  def __init__(self, data_config: ml_collections) -> None:
    self.data_config = data_config

    if self.data_config['name'] == "CIFAR10":
      self.data_loader_class = CIFAR10DataLoader(
          self.data_config['path'], **self.data_config['transforms'])
    elif self.data_config['name'] == 'ReversePredictor':
      self.data_loader_class = ReverseDataLoader(**self.data_config.data_params)
    else:
      self.data_loader_class = None

  def get_train_loader(self) -> data.DataLoader:
    return self.data_loader_class.get_train_loader()

  def get_val_loader(self) -> data.DataLoader:
    return self.data_loader_class.get_val_loader()

  def get_test_loader(self) -> data.DataLoader:
    return self.data_loader_class.get_test_loader()
