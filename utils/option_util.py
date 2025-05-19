import argparse
import yaml
import os
from types import SimpleNamespace # Useful for dot notation access

def load_config():
    parser = argparse.ArgumentParser(description='Load training configuration from YAML.')
    parser.add_argument('-config', type=str, required=True, help='Path to the YAML configuration file.')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {args.config}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {args.config}: {e}")

    config = SimpleNamespace(
        training_args=SimpleNamespace(**config_dict.get('training_args', {})),
        model_params=SimpleNamespace(**config_dict.get('model_params', {}))
    )

    if not hasattr(config.training_args, 'experiment_dir') or not config.training_args.experiment_dir:
        config.training_args.experiment_dir = config.training_args.experiment_name

    return config

