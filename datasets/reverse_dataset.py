from functools import partial
import torch.utils.data as data
import numpy as np


class ReverseDataset(data.Dataset):

  def __init__(self, num_classes, seq_length, num_data_points, np_rng) -> None:
    super().__init__()
    self.nun_classes = num_classes
    self.seq_length = seq_length
    self.num_data_points = num_data_points
    self.np_rng = np_rng

    self.data = self.np_rng.integers(self.nun_classes,
                                     size=(self.num_data_points,
                                           self.seq_length))

  def __len__(self):
    return self.num_data_points

  def __getitem__(self, index):
    inp_data = self.data[index]
    labels = np.flip(inp_data, axis=0)
    return inp_data, labels


# Combine batch elements (all numpy) by stacking
def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)


class ReverseDataLoader:
  """
  Provides train, val and test data loaders for reverse predicton data.
  """
  train_data_loader: data.DataLoader
  val_loader: data.DataLoader
  test_loader: data.DataLoader

  def __init__(self,
               batch_size: int = 128,
               num_classes: int = 10,
               seq_len: int = 16,
               num_train_data_points: int = 50000,
               num_val_data_points: int = 1000,
               num_test_data_points: int = 10000) -> None:
    dataset = partial(ReverseDataset, num_classes, seq_len)
    self.train_loader = data.DataLoader(dataset(
        num_train_data_points, np_rng=np.random.default_rng(42)),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=numpy_collate)
    self.val_loader = data.DataLoader(dataset(num_val_data_points,
                                              np_rng=np.random.default_rng(43)),
                                      batch_size=batch_size,
                                      collate_fn=numpy_collate)
    self.test_loader = data.DataLoader(dataset(
        num_test_data_points, np_rng=np.random.default_rng(44)),
                                       batch_size=batch_size,
                                       collate_fn=numpy_collate)

  def get_train_loader(self) -> data.DataLoader:
    return self.train_loader

  def get_val_loader(self) -> data.DataLoader:
    return self.val_loader

  def get_test_loader(self) -> data.DataLoader:
    return self.test_loader
