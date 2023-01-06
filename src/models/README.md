# Models
This directory contains all the code that defines models and how to train them. The file `model_base.py` defines a base class that all models inherit from. The DeepDeinterlacing model as described in the paper is defined in the file `deep_deinterlacing.py`.

## Loading model parameters
During training the model saves checkpoints in its working directory. The model can be initialized with the trained parameters later using the files in the checkpoint directory. To load the model parameters:
1. Change the `state_dict_path` property in the model config file to the path of the corresponding `model_state_dict.pt` file.
2. Initialize the model with the `_initialize_model_from_config` method with the path to the updated config file.

## Defining a new model
To define a new model one should create a new file and define a class that inherits from ModelBase.
In that file make sure to
1. Define a [configuration class](..\utils\README.md) that inherits from BaseConfig that defines the properties specific to the new model.
2. Define a [configuration class](..\utils\README.md) that inherits from BaseConfig that defines the properties specific to the training procedure of the new model.
3. Decorate the class of the new model with the `register_model` decorator as defined in `model_base.py`. The register model decorator should be provided with a tag that is used to select the model type in config files. The other arguments are to link the model to de model configuration and training configuration classes as defined in step 1, and 2.
4. Define the model architecture by overwriting the `_initialize_architecture` method.

> **Warning**
> Do not overwrite the `__init__` method of the model without calling the parent constructor. Define all variables and model structure in the `_initialize_architecture` method.

> **Warning**
> Do not forget to aggregate submodules in a torch.nn.Modulelist to ensure they are registered.

5. Overwrite the `forward` method that defines how data passes through the model

6. Overwrite the `_forward_and_compute_loss` method that performs a forward pass and computes the training loss.

7. **Optional:** Overwrite the `_create_callbacks` method to add model-specific callbacks. For more on defining callbacks please refer to the [utils README](../utils/README.md). The method should look something like this:
```{Python}
def _create_callbacks(self):
        callbacks = super()._create_callbacks()
        callbacks.append(MyNewCallback(...))
        return callbacks
```
8. Import the model file in `src/__init__.py`