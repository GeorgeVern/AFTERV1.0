import yaml
import os
from sys_config import DATA_DIR, TRAINED_PATH, DEFAULT_OPTS, MODEL_CNF_DIR, CACHED_MODELS_DIR


def load_config(config_file):
    """
    Loads a yaml configuration file
    :param config_file: the absolute path of the yaml file
    :return: the loaded dictionary
    """
    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
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
    trained_model_filename = "After{}/".format(config["model_name_or_path"].split("-")[0].upper())
    config["output_dir"] = "".join((TRAINED_PATH, trained_model_filename, config["task_name"]))
    config["cache_dir"] = CACHED_MODELS_DIR
    default_opts = load_config(defaults_file)
    config.update(default_opts)
    return config
