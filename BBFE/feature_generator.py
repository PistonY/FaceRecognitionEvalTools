__all__ = ['FeatureGenerator']

import torch
import numpy as np
from torch.nn import functional as F
from ..utils import load_model, AlignTransform, _dtypes, get_model_info
from torchvision.transforms import ToTensor


class FeatureGenerator(object):
    """
    This module only support TorchScript model.

    :param param_path: TorchScript model path.
    :param device: device to load.
    :param model_dtype: model datatype only 'float32' and 'float16' are accepted.
    :param output_dtype: convert output to specify dtype.
    :param align: align image by landmark.
    :param flip: use flip added.
    """

    def __init__(self, param_path: str, device='cpu',
                 model_dtype='float32', output_dtype=None,
                 align=True, flip=False):
        assert model_dtype in ('float32', 'float16')

        self.output_dtype = model_dtype if output_dtype is None else output_dtype
        self.device = device
        self.dtype = model_dtype
        self.flip = flip
        self.model = load_model(param_path, mode='script')
        dims, dps = get_model_info(self.model, input_shape=(1, 3, 112, 112), device=device)
        assert dps == _dtypes[model_dtype],\
            f'model actual dtype {dps} but got model_dtype {model_dtype}.'
        print(f'Embedding Size: {dims}')
        self.align_img = AlignTransform() if align else None
        self.to_tensor = ToTensor()

    @torch.no_grad()
    def get_feature(self, image: np.ndarray, bbox: np.ndarray, landmark: np.ndarray):
        assert len(image.shape) == 3 and landmark.shape == (5, 2)
        if self.align_img:
            image = self.align_img(image, landmark)
        image = self.to_tensor(image).unsqueeze(0)
        image = image.to(device=self.device, dtype=_dtypes[self.dtype], non_blocking=True)
        image_flip = image.flip(3) if self.flip else None
        feature = self.model(image)
        feature_flip = self.model(image_flip) if self.flip else None
        feature = F.normalize(feature + feature_flip) if self.flip else feature
        feature = feature.cpu().numpy().flatten().astype(self.output_dtype)
        return feature
