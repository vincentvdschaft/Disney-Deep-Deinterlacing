"""
Base class for models.
New model types inherit from ModelBase and should
- Define a configuration class that inherits from BaseModelConfig
- Define a configuration class that inherits from BaseTrainingConfig
- define an `_initialize_architecture` method
- be decorated with the register_model decorator to link them to a tag and a
model- and training config class
- Overwrite the `forward` method
- Overwrite the `_forward_and_compute_loss` method
- Optional: Overwrite the _create_callbacks method
"""
import torch
from torch import nn
import abc
from pathlib import Path
from tqdm import tqdm
import warnings
try:
    import wandb
except ModuleNotFoundError:
    warnings.warn(
        'module wandb not found. Weights and Biases experiment tracking not possible!')

from src.utils.config import Config
from src.utils.common import create_unique_dir
from src.utils.callback_handler import CallbackHandler, BaseCallback, BasePeriodicCallback
from src.utils.logging import get_main_logger, get_file_handler
from src.datasets import initialize_dataset_from_config


# Dict mapping a tag as used in config files to a model class
_MODEL_FROM_TAG = {}
# Dict mapping a model class to its corresponding model config class
_MODEL_CONFIG = {}
# Dict mapping a model class to its corresponding training config class
_MODEL_TRAINING_CONFIG = {}


def register_model(
        cls=None,
        *,
        model_tag=None,
        model_config_class=None,
        training_config_class=None):
    """
    A decorator for registering model classes, linking it to a model- and
    training-config class as well as to a model tag to use in config files.
    Decorate every new model with this decorator.

    Args:
        model_tag (str): An identifier to use in config files to refer
        to this model.
        model_config_class (:obj:`Config`): The config class to be used to
        initialize this model.
        training_config_class (:obj:`Config`): The config class to be used
        to train this model.

    Returns:
        The input class cls unchanged, but registered
    """

    def _register(cls):
        # Ensure that inputs are supplied
        if model_tag is None:
            raise ValueError('model_tag cannot be None')
        if model_config_class is None:
            raise ValueError('model_config_class cannot be None')
        if training_config_class is None:
            raise ValueError('training_config_class cannot be None')

        # Store the mapping from tag to model class
        _MODEL_FROM_TAG[model_tag.lower()] = cls
        # Store the mapping from model class to config classes
        _MODEL_CONFIG[cls] = model_config_class
        _MODEL_TRAINING_CONFIG[cls] = training_config_class

        # Return model class unchanged
        return cls

    # Return decorator method
    if cls is None:
        return _register
    else:
        return _register(cls)


class ModelBase(nn.Module, abc.ABC):
    """
    Abstract model base class.
    """

    def __init__(self, config, device='cpu'):
        super().__init__()
        # Keeps track of the device the model is on
        self.device = device
        # Initializes config object from file/dict/config
        self.config = self._initialize_config(config)
        #
        self.callback_handler = None
        # Initialize the actiavation function from config
        self.activation = self._get_activation(self.config.activation)
        # Initialize training config to None. Is updated when train() is
        # callded.
        self.training_config = None
        # Dict containing all state needed to restore a training procedure
        self.state = {
            'iteration': 0,
            'batch_loss': [],
            'test_loss': [],
            'lr': []
        }
        # The directory to save checkpoints and such
        self.working_dir = None
        # Keeps track of wether Weights and Biases is enabled
        self.wandb = False

        self.optimizer = None
        self.learningrate_scheduler = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        self._initialize_architecture()
        self.to(device)
        logger = get_main_logger()

        if 'config_path' in self.config:
            state_dict_path = Path(Path(self.config.config_path).parent, 'model_state_dict.pt')
            if state_dict_path.is_file():
                self.config.state_dict_path = state_dict_path
                logger.warning('Loading config from checkpoint folder.'
                ' Changing property state_dict_path to load parameters!')

        # Load model parameters if required
        if self.config.state_dict_path:
            try:
                state_dict = torch.load(self.config.state_dict_path, map_location=self.device)
                self.load_state_dict(state_dict)
                logger.info('Model parameters loaded succesfully')
            except FileNotFoundError:
                logger.warning('Unable to load model parameters')
        else:
            # Apply weight initialization recursively to all submodules
            self.apply(self._init_weights)

    @abc.abstractmethod
    def _initialize_architecture(self):
        pass

    def _init_weights(self, module):
        """
        initializes the weights to normally distributed values with std
        `initializer_range` as specified in the config file. The biases are set
        to zero.
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _initialize_config(self, config):
        """Initializes the Config subclass corresponding to the model.

        Args:
            config (path, Path): Path to config `.yaml` file
            config (dict, `Config`): Configuration dictionary"""
        self.config = _MODEL_CONFIG[self.__class__](config)
        # Return is not needed, but present to show self.config in __init__
        return self.config

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Performs a forward pass through the model."""
        pass

    @abc.abstractmethod
    def _forward_and_compute_loss(self, batch, train=False, **kwargs):
        """Performs a forward pass and computes the loss."""
        pass

    @torch.no_grad()
    def evaluate(self):
        """Evaluates the model by passing all batches in the test set through
        the model and computing the average loss."""

        logger = get_main_logger()

        assert self.test_loader is not None, 'No test dataloader initialized.'

        total_loss = torch.zeros(1, device=self.device)

        for batch in tqdm(self.test_loader):

            # Bring batch to device
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)

            loss = self._forward_and_compute_loss(batch, train=False)

            total_loss += loss

        try:
            return total_loss.item() / len(self.test_loader)
        except ZeroDivisionError:
            logger.warning(
                'There are zero batches in the test loader!')
            return torch.zeros(1, device=self.device)

    def _register_standardization_buffer(self, num_input_channels):
        """
        Registers buffers to store mu and sigma and initializes with all 0 and 1.
        """
        self.register_buffer(
            name='standardization_mu',
            tensor=torch.zeros(num_input_channels, device=self.device))
        self.register_buffer(
            name='standardization_sigma',
            tensor=torch.ones(num_input_channels, device=self.device))

    def reverse_standardization(self, batch):
        """
        Reverses the standardization that the model was trained with to get back
        to the real data statistics.
        Parameters
        ----------
        batch : tensor (B, C, H, W)
            The tensor produced by the network
        Returns
        -------
        tensor (B, C, H, W)
            Unstandardized tensor
        """
        num_channels = self.standardization_mu.shape[0]
        mu = self.standardization_mu.reshape(1, num_channels, 1, 1)
        sigma = self.standardization_sigma.reshape(1, num_channels, 1, 1)

        return (batch * sigma) + mu

    def apply_standardization(self, batch):
        """Applies standardization to batch."""
        num_channels = self.standardization_mu.shape[0]
        mu = self.standardization_mu.view(1, num_channels, 1, 1)
        sigma = self.standardization_sigma.view(1, num_channels, 1, 1)
        return (batch - mu) / sigma

    def count_parameters(self):
        """Counts all trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_activation(self, activation_type, slope=0.1):
        """Returns a callable activation function of the desired type.

        Args:
            activation_type: One of {'relu', 'leaky_relu', 'sigmoid', 'swish'}
            slope: The negative slope (only used by leaky ReLU)

        Returns:
            A callable activation function
        """
        # Define mapping from name to scheduler class
        str_to_act = {
            'relu': torch.nn.functional.relu,
            'leaky_relu': lambda x: torch.nn.functional.leaky_relu(x, slope),
            'sigmoid': torch.nn.functional.sigmoid,
            'tanh': torch.nn.functional.tanh,
            'swish': torch.nn.functional.silu,
            'silu': torch.nn.functional.silu
        }
        # Try to find the class corresponding to the given name
        try:
            activation_function = str_to_act[activation_type.lower()]
        except KeyError as exc:
            raise NotImplementedError(
                'ERROR: Invalid activation function. Choose'
                f'an activation function from {list(str_to_act.keys())}.') from exc

        return activation_function

    def train_model(self, training_config):
        """Trains the model based on the provided training config file. This
        method calls the methods `_prepare_for_training_loop` and
        `_training_loop`. These can be overwritten or extended by subclasses.
        """
        # Store training_config internally
        self.training_config = _MODEL_TRAINING_CONFIG[self.__class__](
            training_config)

        self._initialize_wandb()
        self._prepare_for_training_loop()
        self._training_loop()

    def _initialize_wandb(self):
        """Initializes Weights and Biases if `wandb` is present and set to True
        in config. The run is stored in the internal state.

        Returns:
            Run as returned by wandb.init()
        """
        training_config = self.training_config
        if 'wandb' in training_config and training_config.wandb is True:

            # Aggregate configs in wandb_config to upload
            wandb_config = {
                'model_config': self.config,
                'training_config': self.training_config
            }

            # Initialize Weights and Biases
            run = wandb.init(config=wandb_config,
                             project=training_config.wandb_project_name)

            self.state['run'] = run
            self.wandb = True

            return run

    def _prepare_for_training_loop(self):

        # Create working directory for when using wandb with run name
        if 'run' in self.state:
            dirname = self.state['run'].name
            project = self.state['run'].project
            self.working_dir = Path('models', project, dirname)
            self.working_dir.mkdir(parents=True)
        # Create working directory with name of model otherwise
        else:
            dirname = self.config.model_class
            self.working_dir = create_unique_dir(
                parent_directory=Path('models', self.config.model_class),
                name=dirname)
        
        logger = get_main_logger()
        # Add handler to print to file
        logger.addHandler(get_file_handler(self.working_dir, 'training_log'))

        # Save configs
        self.config.save_to_yaml(Path(self.working_dir, 'model_config.yaml'))
        self.training_config.save_to_yaml(
            Path(self.working_dir, 'training_config.yaml'))

        self.optimizer = initialize_optimizer(
            self.training_config.optimizer,
            self.training_config.optimizer_parameters,
            self.parameters())

        self.state['lr'].append(self.optimizer.param_groups[0]['lr'])

        self.state['optimizer_state_dict'] = self.optimizer.state_dict()

        self.learningrate_scheduler = initialize_scheduler(
            optimizer=self.optimizer,
            scheduler_tag=self.training_config.scheduler,
            scheduler_parameters=self.training_config.scheduler_parameters)

        # Initialize dataset
        self.train_dataset = initialize_dataset_from_config(
            self.training_config.dataset_config,
            train=True)
        self.dataset_config = self.train_dataset.config

        self.test_dataset = initialize_dataset_from_config(
            self.training_config.dataset_config,
            train=False)

        self.train_loader = self.train_dataset.get_dataloader(
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            limit_num_samples=self.training_config.train_limit_num_samples)

        self.test_loader = self.test_dataset.get_dataloader(
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            limit_num_samples=self.training_config.test_limit_num_samples)

        # Initialize callback handler
        callbacks = self._create_callbacks()
        self.callback_handler = CallbackHandler(callbacks)

    def _training_loop(self):

        self.callback_handler.before_first_iteration()

        logger = get_main_logger()

        training_config = self.training_config

        iterator = iter(self.train_loader)

        while self.state['iteration'] < training_config.num_iterations:

            # Get a new batch from the iterator and reset the iterator if
            # needed
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch = next(iterator)

            # Bring batch to device
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)

            loss = self._forward_and_compute_loss(batch, train=False)

            

            if self.state['iteration'] % 25 == 0:
                logger.info('%i - batch loss: %f',
                            self.state['iteration'], loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            # Clip the gradients
            if self.training_config.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.training_config.gradient_clip_norm)

            self.optimizer.step()

            self.state['batch_loss'].append(loss.item())
            self.state['iteration'] += 1

            self.callback_handler.iteration_end(
                iteration=self.state['iteration'])

    def _create_callbacks(self):
        callbacks = []

        # Weights and Biases logging callback
        if self.wandb:
            callbacks.append(
                LogToWandBCallback(whens=['iteration_end'], model=self))

        # Model testset evaluation callback
        callbacks.append(EvaluateModelCallback(
            period=self.training_config.evaluation_interval,
            periodic_whens=['iteration_end'],
            unconditional_whens=['before_first_iteration'],
            model=self))

        # Makes a step with the learningrate scheduler
        if self.learningrate_scheduler is not None:
            callbacks.append(UpdateSchedulerCallback(
                period=self.training_config.scheduler_interval,
                periodic_whens=['iteration_end'],
                unconditional_whens=None,
                model=self))

        # Saves a checkpoint after checkpoint_interval
        callbacks.append(SaveCheckpointCallback(
            period=self.training_config.checkpoint_interval,
            periodic_whens=['iteration_end'],
            unconditional_whens=None,
            model=self))

        # Saves checkpoint every 1000 iterations overwriting check_latest
        callbacks.append(SaveCheckpointCallback(
            period=1000,
            periodic_whens=['iteration_end'],
            unconditional_whens=None,
            model=self,
            overwrite_latest=True))

        return callbacks


class ModelConfig(Config):
    """Model config super class."""

    def _validate(self):
        super()._validate()
        self._must_contain('model_class', element_type=str)
        self._must_contain('state_dict_path', element_type=(str, bool))
        self._must_contain('activation', element_type=str)


class TrainingConfig(Config):
    """Model config super class."""

    def _validate(self):
        super()._validate()
        self._must_contain('model_config', element_type=str)
        self._must_contain('dataset_config', element_type=str)
        self._must_contain('train_limit_num_samples',
                           element_type=int, larger_than=0)
        self._must_contain('test_limit_num_samples',
                           element_type=int, larger_than=0)
        self._must_contain('num_iterations', element_type=int, larger_than=0)
        self._must_contain('batch_size', element_type=int, larger_than=0)
        self._must_contain('evaluation_interval',
                           element_type=int, larger_than=0)
        self._must_contain('checkpoint_interval',
                           element_type=int, larger_than=0)
        self._must_contain('optimizer', element_type=str)
        self._must_contain('optimizer_parameters')
        self._must_contain('gradient_clip_norm')
        self._must_contain('scheduler')
        self._must_contain('scheduler_parameters')
        self._must_contain('scheduler_interval', element_type=int,
                           larger_than=0)

        # Check wandb settings if present, but allow ommission
        if 'wandb' in self:
            self._must_contain('wandb', element_type=bool)
            # Only require project name when actually using wandb
            if self.wandb is True:
                self._must_contain('wandb_project_name', element_type=str)


class LogToWandBCallback(BaseCallback):
    """Callback to log data to wandb."""

    def __init__(self, whens, model):
        super().__init__(whens)
        self.model = model

    def _trigger(self, when, **kwargs):
        wandb.log({
            'batch loss': self.model.state['batch_loss'][-1],
            'lr': self.model.state['lr'][-1]
        },
            step=self.model.state['iteration'])


class EvaluateModelCallback(BasePeriodicCallback):
    def __init__(self, period, periodic_whens, unconditional_whens, model):
        super().__init__(period, periodic_whens, unconditional_whens)
        self.model = model

    def _trigger(self, when, **kwargs):
        logger = get_main_logger()
        logger.info('Evaluating model...')
        test_loss = self.model.evaluate()

        self.model.state['test_loss'].append(test_loss)
        if self.model.wandb:
            wandb.log(
                {'test loss': self.model.state['test_loss'][-1]},
                step=self.model.state['iteration'],
                commit=False)
        logger.info('test loss: %f', test_loss)


class SaveCheckpointCallback(BasePeriodicCallback):
    """Callback to store a checkpoint to the working directory"""

    def __init__(
            self,
            period,
            periodic_whens,
            unconditional_whens,
            model,
            overwrite_latest=False):
        super().__init__(period, periodic_whens, unconditional_whens)
        self.model = model
        self.overwrite_latest = overwrite_latest

    def _trigger(self, when, **kwargs):
        # Create output checkpoint directory
        iteration = self.model.state['iteration']
        if not self.overwrite_latest:
            checkpoint_dir = Path(self.model.working_dir, 'checkpoints',
                                  'checkpoint_' + str(iteration).zfill(8))
            checkpoint_dir.mkdir(parents=True)
        else:
            checkpoint_dir = Path(self.model.working_dir, 'checkpoints',
                                  'checkpoint_latest')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Store the model parameters
        path = Path(checkpoint_dir, 'model_state_dict.pt')
        torch.save(self.model.state_dict(), path)

        # Store the training state (optimizer state_dict)
        path = Path(checkpoint_dir, 'training_state.pt')
        torch.save(self.model.state, path)

        # Store config files
        for config, name in [
            (self.model.config, 'model_config'),
            (self.model.training_config, 'training_config'),
            (self.model.dataset_config, 'dataset_config')
        ]:
            path = Path(checkpoint_dir, name + '.yaml')
            config.save_to_yaml(path)

        logger = get_main_logger()
        logger.info(
            'Checkpoint saved to \x1b[1;36;20m %s \x1b[0m',
            str(checkpoint_dir))


class UpdateSchedulerCallback(BasePeriodicCallback):
    """Callback to make a step with the learningrate scheduler."""

    def __init__(self, period, periodic_whens, unconditional_whens, model):
        super().__init__(period, periodic_whens, unconditional_whens)
        self.model = model

    def _trigger(self, when, **kwargs):

        # Only insert the loss as an argument for the scheduler that requires
        # it
        if isinstance(
                self.model.learningrate_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau):
            step_args = (self.model.state['test_loss'][-1],)
        else:
            step_args = ()
        self.model.learningrate_scheduler.step(*step_args)
        get_main_logger().info(
            'lr: %f', self.model.optimizer.param_groups[0]['lr'])
        self.model.state['lr'].append(self.model.optimizer.param_groups[0]['lr'])


def initialize_optimizer(
        optimizer_tag,
        optimizer_parameters,
        trainable_parameters):
    """Initializes an optimizer from a tag with the given parameters.

    Args:
        optimizer_tag (str): Choose from ('adam', 'adadelta', 'adagrad',
        'adamw', 'sparseadam', 'adamax', 'asgd', 'lbfgs', 'rmsprop', 'sgd')
        optimizer_parameters (dict): The keyword arguments to pass to optimizer
        init trainable_parameters (state dict): The trainable parameters to
        optimize as retrieved by model.parameters()

    Returns:
        torch.optim.optimizer object.
    """

    tag_to_opt = {
        'adam': torch.optim.Adam,
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adamw': torch.optim.AdamW,
        'sparseadam': torch.optim.SparseAdam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'lbfgs': torch.optim.LBFGS,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD
    }
    # Try to find the class corresponding to the given name
    try:
        optimizer_class = tag_to_opt[optimizer_tag.lower()]
    except KeyError as exc:
        raise NotImplementedError(f'WARNING: Invalid optimizer. \
        Choose an optimizer from {list(tag_to_opt.keys())}.') from exc

    if optimizer_parameters is None:
        optimizer_parameters = {}

    optimizer = optimizer_class(trainable_parameters, **optimizer_parameters)

    return optimizer


def initialize_scheduler(scheduler_tag, scheduler_parameters, optimizer):
    """Initializes an scheduler from a tag with the given parameters.

    Args:
        scheduler_tag (str): Choose from ('lambda', 'multiplicative', 'step',
        'constant', 'linear', 'plateau') scheduler_parameters (dict): The
        keyword arguments to pass to scheduler init trainable_parameters (state
        dict): The trainable parameters to optimize as retrieved by
        model.parameters()

    Returns:
        torch.optim.scheduler object or None if scheduler_tag is False
    """

    if scheduler_tag is False:
        return None

    tag_to_sched = {
        'lambda': torch.optim.lr_scheduler.LambdaLR,
        'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
        'step': torch.optim.lr_scheduler.StepLR,
        'constant': torch.optim.lr_scheduler.ConstantLR,
        'linear': torch.optim.lr_scheduler.LinearLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
    }
    # Try to find the class corresponding to the given name
    try:
        scheduler_class = tag_to_sched[scheduler_tag.lower()]
    except KeyError as exc:
        raise NotImplementedError(f'WARNING: Invalid scheduler. \
        Choose an scheduler from {list(tag_to_sched.keys())}.') from exc

    if scheduler_parameters is None:
        scheduler_parameters = {}

    scheduler = scheduler_class(optimizer=optimizer, **scheduler_parameters)

    return scheduler
