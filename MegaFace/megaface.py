import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Eval.MegaFace.utils import MegafaceDataset, write_bin
from Eval.utils import check_dir, load_model, l2norm, get_model_info

parser = argparse.ArgumentParser(description='Generate MegaFace eval files.')

parser.add_argument('--model-path', type=str, required=True,
                    help='only torchscript model is accept.')
parser.add_argument('--megaface-root', type=str, default='',
                    help='megaface root dir.')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--remove-noise', action='store_true',
                    help='remove noise by given list.')
parser.add_argument('--flip', action='store_true',
                    help='use flip added test.')

args = parser.parse_args()
np.random.seed(1024)

gen_path = lambda prefix: os.path.join(args.megaface_root, prefix)
model_name = args.model_path.split('/')[-1].split('.')[0]
print(f'Model name: {model_name}.')


def get_input_path():
    slp = gen_path('facescrub_lst')
    srp = gen_path('facescrub_images')
    mlp = gen_path('megaface_lst')
    mrp = gen_path('megaface_images')

    return slp, srp, mlp, mrp


def load_noise_list():
    mnl = gen_path('megaface_noises.txt')
    snl = gen_path('facescrub_noises.txt')
    with open(snl, 'r') as f:
        s_meta = f.read().splitlines()[1:]
    with open(mnl, 'r') as f:
        m_meta = f.read().splitlines()[1:]
        m_meta = [file.split('/')[-1] for file in m_meta]
    return s_meta, m_meta


@torch.no_grad()
def write_generated_features(lst_path, root_path, output_root, model,
                             dtype, flip=False, noise=None):
    trans = transforms.ToTensor()
    val_data = MegafaceDataset(lst_path, root_path, transform=trans)
    feature_names = val_data.get_file_lst()
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
    device = torch.device("cuda:0")
    for data, items in tqdm(val_loader):
        data = data.to(device, dtype, non_blocking=True)
        data_flip = data.flip(3) if flip else None
        embeddings = model(data)
        embeddings_flip = model(data_flip) if flip else None
        embeddings = F.normalize(embeddings + embeddings_flip) if flip else embeddings
        for embedding, idx in zip(embeddings, items):
            name = feature_names[int(idx)].split('/')[-1]
            folder = feature_names[int(idx)][:-len(name)]
            check_dir(os.path.join(output_root, folder))
            if noise is not None and name in noise:
                embedding = l2norm(np.random.random((embedding_size,)))
            else:
                embedding = embedding.cpu().numpy().astype(np.float32)

            write_bin(os.path.join(output_root, folder, f'{name}_{model_name}.bin'),
                      embedding)


if __name__ == '__main__':
    rm_noise = args.remove_noise
    facescrub_out = os.path.join(args.output_root, 'facescrub')
    megaface_out = os.path.join(args.output_root, 'MegaFace')
    check_dir(facescrub_out, megaface_out)
    slp, srp, mlp, mrp = get_input_path()
    s_noise, m_noise = None, None
    if rm_noise:
        s_noise, m_noise = load_noise_list()

    model = load_model(args.model_path, device='cuda:0', mode='script', parallel=True)
    embedding_size, dtype = get_model_info(model, (1, 3, 112, 112), device='cuda:0')
    print(f'Model Info: embedding size: {embedding_size}, datatype: {dtype}.')

    print('Generating facescrub features.')
    write_generated_features(slp, srp, facescrub_out, model, dtype, args.flip, s_noise)
    print('Generating MegaFace features.')
    write_generated_features(mlp, mrp, megaface_out, model, dtype, args.flip, m_noise)
