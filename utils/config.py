import yaml
import os
from sys_config import DATA_DIR, TRAINED_PATH, DEFAULT_OPTS, MODEL_CNF_DIR


def load_config(config_file):
    """
    Loads a yaml configuration file
    :param config_file: the absolute path of the yaml file
    :return: the loaded dictionary
    """
    with open(config_file) as file:
        cfg = yaml.load(file)
    return cfg


def train_options(config_file, defaults_file=DEFAULT_OPTS):
    """
    Reads a yaml file and combines it with the default input parameters
    :param config_file: input configuration yaml
    :param defaults_file: yaml file with default parameters
    :return: updated configuration yaml
    """
    config = load_config(os.path.join(MODEL_CNF_DIR, config_file))
    config["data_dir"] = "".join((DATA_DIR, config["task_name"]))
    config["output_dir"] = "".join((TRAINED_PATH, config["task_name"]))
    default_opts = load_config(defaults_file)
    config.update(default_opts)
    return config
