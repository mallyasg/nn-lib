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


dataset = partial(ReverseDataset, 10, 16)
rev_train_loader = data.DataLoader(dataset(50000,
                                           np_rng=np.random.default_rng(42)),
                                   batch_size=128,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=numpy_collate)
rev_val_loader = data.DataLoader(dataset(1000,
                                         np_rng=np.random.default_rng(43)),
                                 batch_size=128,
                                 collate_fn=numpy_collate)
rev_test_loader = data.DataLoader(dataset(10000,
                                          np_rng=np.random.default_rng(44)),
                                  batch_size=128,
                                  collate_fn=numpy_collate)
