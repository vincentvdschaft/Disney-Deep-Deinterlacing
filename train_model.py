import torch
import argparse
from src.utils.config import load_config_from_yaml
from src.utils.logging import get_main_logger
import src.models

DESCRIPTION = 'Trains a model based on the parameters in a config file.'

parser = argparse.ArgumentParser(
    description=DESCRIPTION,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    'config_file',
    nargs="?",
    default=r'config\training_config.yaml',
    type=str,
    help="Path to configuration yaml-file to configure the training session")

parser.add_argument(
    "--gpu",
    default=0,
    type=int,
    choices=(0, 1, 2, 3, 4, 5, 6, 7),
    help="The gpu to train on")


logger = get_main_logger()

# Parse arguments
args = parser.parse_args()

logger.info('Running script: Train model')

# Load config file
training_config = load_config_from_yaml(args.config_file)

# Set device
device = torch.device(
    f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
logger.info('Running on %s', device)

model = src.models.initialize_model_from_config(training_config.model_config, device=device)
model.train_model(training_config)
