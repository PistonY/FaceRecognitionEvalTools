import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from utils import load_model
import torch

print(torch.__version__)
model = load_model('../mobilefacenet_glint360k.pt',
                   dtype='float16', mode='script', parallel=True)
