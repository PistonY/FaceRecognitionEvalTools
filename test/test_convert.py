import os
import sys

sys.path.append(os.getcwd())

from utils import export_ts

if __name__ == '__main__':
    export_ts('GhostNet', '/media/devin/data/PycharmProjects/VideoDNA/face_params/GhostNet_Glint360K_56.pt',
              (1, 3, 112, 112), '/media/devin/data/PycharmProjects/VideoDNA/Eval/models/mobilefacenet_glint360k_arcface.pt',
              dtype='float16', remove_module=True, test_ts_model=True)

