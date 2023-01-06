import torch
from .model_base import ModelConfig, _MODEL_FROM_TAG
from pathlib import PurePath, Path
# All models must be imported here to register them
import src.models.deep_deinterlacing

def initialize_model_from_config(config, device='cpu'):
    """Initializes a model from a config file path or a configuration dict.

    Args:
        config (dict, Config): The parameters to initialize the model with.
        config (str, Path): A path to a .yaml config file containing the
        parameters to initialize the model with.

    Returns:
        The initialized model
    """

    # Load config file as generic model config
    config_object = ModelConfig(config)

    # Store config path in config if the method is called with a path
    if isinstance(config, (str, PurePath)):
        config_object.config_path = str(config)

    try:
        # Retrieve the corresponding model class
        model_class = _MODEL_FROM_TAG[config_object.model_class.lower()]

    # Raise error if model type does not exist
    except KeyError as exc:
        raise NotImplementedError(
            (f'Model {config_object.model_class} has not been implemented. Please'
             f'choose from ({_MODEL_FROM_TAG.keys()})')) from exc

    # Initialize the model
    model = model_class(config_object, device=device)

    return model