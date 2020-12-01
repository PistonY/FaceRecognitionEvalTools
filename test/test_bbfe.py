import os
import sys

sys.path.append(os.getcwd())

from BBFE import FeatureGenerator
import numpy as np
import cv2

FG = FeatureGenerator('../models/mobilefacenet_glint360k_arcface.pt',
                      device='cuda:0', model_dtype='float16', align=True, flip=False)

lk = np.array([46.060, 62.026, 87.785, 60.323, 68.851, 77.656, 52.162, 99.875, 86.450, 98.648]).reshape((5, 2))
img = cv2.imread('/media/devin/data/face/face_test/IJB_release/IJBB/loose_crop/1.jpg', cv2.IMREAD_COLOR)[..., ::-1]
feats = FG.get_feature(img, None, lk)
print(feats.shape)
