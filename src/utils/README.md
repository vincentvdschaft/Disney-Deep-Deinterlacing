# Utils
This folder contains modules with helper methods such as logging.

## Logging
The `get_main_logger` method creates a logger that writes all logs to a file and to the stdout. If the logger already exists it just returns it. This way we can call `get_main_logger` anywhere in the code and start pushing log messages.

## Config
The config system works as follows. There is `Config` base class which is a dictionary with some extra features such as dot-indexing and saving to YAML. The `Config` class defines a `_validate` method that checks if its contents are valid. One can now define new specific derived classes that define their own `_validate` methods to specify all kinds of configuration files. Most checks can be performed using the `_must_contain` method. To also check all requirements from the superclass just call the parent method in the child version of `_validate`. This way we can for instance create a specific configuration class that checks everything that is required for all model configurations and also some extra properties that only one model has without copying any code.
```{Python}
class MyConfig(BaseConfig):
    def _validate(self):
        super()._validate()
        self._must_contain('some_variable', type=int)
```
> **Warning**
> Do not overwrite the `__init__` method of the configuration class without calling the parent constructor.

## Callback handler
The callback handler is a system to easily schedule certain tasks at certain points in the code without cluttering up the code. The CallbackHandler object is initialized with a number of Callbacks. These all contain a property `whens` which are the moments that this callback should be called. This could be *iteration_end* for instance. The callback handler then adds a callable attribute to itself. You can now call `callbackhandler.iteration_end()` at the desired location in your code and the callback handler will trigger all callbacks with *iteration_end* in their respecitve list of `whens`.

To add a custom callback, inherit from one of the base callback classes and overwrite the `_trigger` method to define what it should do.

Optionally you can also overwrite the `_trigger_condition` method to add some condition that has to be satisfied to trigger.