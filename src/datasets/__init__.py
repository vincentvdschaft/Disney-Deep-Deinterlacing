import torch
from .datasets import DatasetConfig, _DATASET_FROM_TAG
import src.datasets.vimeo_triplet_dataset


def initialize_dataset_from_config(config, train=False, shuffle=True, device='cpu'):
    """Initializes a dataset from a config file path or a configuration dict.

    Args:
        config (dict, Config): The parameters to initialize the dataset with.
        config (str, Path): A path to a .yaml config file containing the
        parameters to initialize the dataset with.

    Returns:
        The initialized dataset
    """
    # Load config file as generic dataset config
    config = DatasetConfig(config)

    assert not config is None, 'Failed to load config file'

    try:
        # Retrieve the corresponding dataset class
        dataset_class = _DATASET_FROM_TAG[config.dataset_class.lower()]

    # Raise error if dataset type does not exist
    except KeyError as exc:
        raise NotImplementedError(
            (f'Dataset {config.dataset_class} has not been implemented. Please'
             f'choose from ({_DATASET_FROM_TAG.keys()})')) from exc

    # Initialize the dataset
    dataset = dataset_class(config, train=train, shuffle=shuffle, device=device)

    return dataset
