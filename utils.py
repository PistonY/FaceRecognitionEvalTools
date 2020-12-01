__all__ = ['check_dir', 'load_model', 'l2norm', 'AlignTransform',
           'get_model_info']

import os
import cv2
import torch
import numpy as np
import logging
from torch import nn
from skimage import transform as trans

try:
    import models
except ImportError:
    logging.warning('No models found,'
                    'meta load has been abandoned,'
                    'only script load is accepted.')
    models = None

_dtypes = {
    'float32': torch.float,
    'float16': torch.half
}


def check_dir(*path):
    """Check dir(s) exist or not, if not make one(them).
    Args:
        path: full path(s) to check.
    """
    for p in path:
        os.makedirs(p, exist_ok=True)


def load_model(param_path: str, name=None, device='cuda:0',
               dtype='float32', mode='meta', parallel=False, **kwargs):
    """
    Load model from pytorch meta file or load a script model.


    :param name: model name from model_zoo, script model will ignore this.
    :param param_path: param path if load from meta, model file if load a script model.
    :param device: model will be loaded on cpu and then move to device.
    :param dtype:  model dtype, only FP32 and FP16 supported. Only support meta load.
    :param mode: mode to load a model.
    :param parallel: whether return a parallel model.
    :param kwargs:
    :return:
    """
    assert mode in ('meta', 'script')
    assert dtype in ('float16', 'float32')
    device = torch.device(device)
    if mode == 'meta':
        assert models is not None
        model = models.__dict__[name](**kwargs).to(_dtypes[dtype])
        model = nn.DataParallel(model)
        checkpoint = torch.load(param_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif mode == 'script':
        model = torch.jit.load(param_path, map_location='cpu')
    else:
        raise NotImplementedError

    if dtype != 'float32' and device.type == 'cpu':
        logging.warning(f"Some operations in {dtype} may not support {device}, "
                        f"better load model to a cuda device.")
    model = model.to(device)
    if parallel:
        assert device.type == 'cuda'
        model = nn.DataParallel(model)
    return model


@torch.no_grad()
def export_ts(name: str, param_path: str, input_shape: tuple,
              save_path: str, dtype='float32', remove_module=False,
              test_ts_model=False, **kwargs):
    """
    Load a pytorch meta model and convert to TorchScript model.

    :param name: model name,
    :param param_path:  param path.
    :param input_shape: model inference input shape.
    :param save_path: path to save script model.
    :param dtype: model dtype.
    :param remove_module: Remove module if model train using DP or DDP.
    :param test_ts_model: Test te model have same output with torch model.
    :return: TorchScript model.
    """

    device = 'cpu' if dtype == 'float32' else 'cuda:0'
    model = load_model(param_path, name, device=device, dtype=dtype, mode='meta', **kwargs)
    if remove_module:
        model = model.module
    model.eval()
    x = torch.rand(*input_shape, dtype=_dtypes[dtype], device=device)
    traced_script_module = torch.jit.trace(model, x)
    if test_ts_model:
        test_x = torch.rand(*input_shape, dtype=_dtypes[dtype], device=device)
        assert torch.sum(model(test_x) - traced_script_module(test_x)) == 0
    torch.jit.save(traced_script_module, save_path)


def l2norm(arr: np.ndarray):
    return arr / np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)


def get_model_info(model, input_shape: tuple, device: str):
    """
    Get embedding_size and dtype of a model.
    :param model: model to test.
    :param input_shape: model inference input shape.
    :param device: model device(cuda or cpu.)
    :return: embedding_size, dtype
    """

    test_x = torch.rand(*input_shape, dtype=_dtypes['float32'], device=device)
    try:
        y = model(test_x)
    except RuntimeError:
        y = model(test_x.half())

    return y.size(1), y.dtype


class AlignTransform(nn.Module):
    # Align img by given landmark.
    src = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.729904, 92.2041]], dtype=np.float32)

    def __init__(self):
        super().__init__()
        # src = np.array([
        #     [30.2946, 51.6963],
        #     [65.5318, 51.5014],
        #     [48.0252, 71.7366],
        #     [33.5493, 92.3655],
        #     [62.7299, 92.2041]], dtype=np.float32)
        # src[:,0] += 8
        self.tform = trans.SimilarityTransform()
        self.image_size = (112, 112)

    def forward(self, img: np.ndarray, landmark: np.ndarray):
        assert landmark.shape == (5, 2)
        self.tform.estimate(landmark, self.src)
        M = self.tform.params[0:2, :]
        img = cv2.warpAffine(img, M, self.image_size, borderValue=0.0)
        return img
