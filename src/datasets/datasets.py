from pathlib import Path
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
import abc
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from src.utils.common import *
from src.utils.config import Config

_DATASET_FROM_TAG = {}
_DATASET_CONFIG = {}

def register_dataset(
        cls=None,
        *,
        dataset_tag=None,
        dataset_config_class=None):
    """
    A decorator for registering dataset classes, linking it to a dataset
    config class as well as to a model tag to use in config files.
    Decorate every new dataset with this decorator.

    Args:
        dataset_tag (str): An identifier to use in config files to refer
        to this dataset.
        dataset_config_class (:obj:`Config`): The config class to be used to
        initialize this dataset.

    Returns:
        The input class cls unchanged, but registered
    """

    def _register(cls):
        # Ensure that inputs are supplied
        if dataset_tag is None:
            raise ValueError('model_tag cannot be None')
        if dataset_config_class is None:
            raise ValueError('model_config_class cannot be None')

        # Store the mapping from tag to model class
        _DATASET_FROM_TAG[dataset_tag.lower()] = cls
        # Store the mapping from model class to config classes
        _DATASET_CONFIG[cls] = dataset_config_class

        # Return model class unchanged
        return cls

    # Return decorator method
    if cls is None:
        return _register
    else:
        return _register(cls)

class DatasetConfig(Config):
    """Base dataset config class. Further requirements will be added in child
    classes."""
    def _validate(self):
        self._must_contain('dataset_class')


class Dataset(abc.ABC):
    """Dataset base class."""
    def __init__(self, config):
        # Initializes config object from file/dict/config
        self.config = self._initialize_config(config)

    def _initialize_config(self, config):
        """Initializes the Config subclass corresponding to the dataset.

        Args:
            config (path, Path): Path to config `.yaml` file
            config (dict, `Config`): Configuration dictionary"""
        self.config = _DATASET_CONFIG[self.__class__](config)
        # Return is not needed, but present to show self.config in __init__
        return self.config

    def get_dataloader(self, batch_size=64, shuffle=True, num_workers=0, limit_num_samples=False):
        """Generates a dataloader from the dataset."""

        if batch_size > len(self):
            raise RuntimeError('Batch size cannot be larger than dataset size.')

        if not limit_num_samples is False:
            dataset = torch.utils.data.Subset(self, range(limit_num_samples))
        else:
            dataset = self

        return torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers)
