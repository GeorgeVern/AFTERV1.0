import os
import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINED_PATH = "/data/data1/users/gvernikos/After_v1.0/"

MODEL_CNF_DIR = os.path.join(BASE_DIR, "configs")

DATA_DIR = "/data/data1/users/gvernikos/Datasets/"

DEFAULT_OPTS = os.path.join(BASE_DIR, "configs/default_opts.yaml")
