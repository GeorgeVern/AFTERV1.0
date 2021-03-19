import os
import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINED_PATH = BASE_DIR

# Path to store the configuration files for the experiments
MODEL_CNF_DIR = os.path.join(BASE_DIR, "configs")

# Path to store the dataset files
DATA_DIR = os.path.join(BASE_DIR, "Datasets/")

# Path to store the pre-trained models downloaded from s3
CACHED_MODELS_DIR = os.path.join(BASE_DIR, "cached_models")

DEFAULT_OPTS = os.path.join(BASE_DIR, "configs/default_opts.yaml")
