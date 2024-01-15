import numpy as np
from functools import partial

import torch
import torch.utils.data as data

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


def image_to_numpy(img, mean, std):
  img = np.array(img, dtype=np.float32)
  img = (img / 255. - mean) / std
  return img


# We need to stack the batch elements
def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)


class CIFAR10DataLoader:
  """
  Provides train, val and test data loaders for CIFAR10 dataset.
  """
  train_loader: data.DataLoader
  val_loader: data.DataLoader
  test_loader: data.DataLoader

  def __init__(
      self,
      data_path: str,
      batch_size: int = 128,
      normalize_image: bool = True,
      image_mean: list = [0.49139968, 0.48215841, 0.44653091],
      image_std: list = [0.24703223, 0.24348513, 0.26158784],
      perform_random_horizontal_flip: bool = True,
      random_resized_crop: bool = True,
      random_resized_crop_size: tuple = (32, 32),
      random_resized_crop_scale: tuple = (0.8, 1.0),
      random_resized_crop_ratio: tuple = (0.9, 1.1)
  ) -> None:
    train_transform = []
    if perform_random_horizontal_flip:
      train_transform.append(transforms.RandomHorizontalFlip())

    if random_resized_crop:
      train_transform.append(
          transforms.RandomResizedCrop(random_resized_crop_size,
                                       scale=random_resized_crop_scale,
                                       ratio=random_resized_crop_ratio))

    if normalize_image:
      image_to_numpy_partial = partial(image_to_numpy,
                                       mean=np.array(image_mean),
                                       std=np.array(image_std))
    else:
      image_to_numpy_partial = partial(image_to_numpy,
                                       mean=np.array([0.0, 0.0, 0.0],
                                                     std=np.array(
                                                         [1.0, 1.0, 1.0])))

    train_transform.append(image_to_numpy_partial)
    train_transform = transforms.Compose(train_transform)
    test_transform = image_to_numpy_partial

    train_dataset = CIFAR10(root=data_path,
                            train=True,
                            transform=train_transform,
                            download=True)
    val_dataset = CIFAR10(root=data_path,
                          train=True,
                          transform=test_transform,
                          download=True)

    train_set, _ = data.random_split(
        train_dataset, [45000, 5000],
        generator=torch.Generator().manual_seed(42))

    _, val_set = torch.utils.data.random_split(
        val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    test_set = CIFAR10(root=data_path,
                       train=False,
                       transform=test_transform,
                       download=True)

    # We define a set of data loaders that we can use for training and validation
    self.train_loader = data.DataLoader(train_set,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=numpy_collate,
                                        num_workers=8,
                                        persistent_workers=True)

    self.val_loader = data.DataLoader(val_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      collate_fn=numpy_collate,
                                      num_workers=4,
                                      persistent_workers=True)

    self.test_loader = data.DataLoader(test_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       collate_fn=numpy_collate,
                                       num_workers=4,
                                       persistent_workers=True)

  def get_train_loader(self) -> data.DataLoader:
    return self.train_loader

  def get_val_loader(self) -> data.DataLoader:
    return self.val_loader

  def get_test_loader(self) -> data.DataLoader:
    return self.test_loader
