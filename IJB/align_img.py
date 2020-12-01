import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import cv2
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import check_dir
from IJB.utils import IJBDataset

parser = argparse.ArgumentParser(description='Align IJB Images.')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--root-path', required=True, type=str, help='Data root path.')
parser.add_argument('--retina-landmark', dest='rl', action='store_true', help='use retina face landmark.')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=0)
args = parser.parse_args()

target_path = os.path.join(args.root_path, args.target)
img_path = os.path.join(target_path, 'loose_crop')
lst_path = os.path.join(target_path, 'meta',
                        f"{args.target.lower()}_name_5pts_score{'_retina' if args.rl else ''}.txt")
save_path = os.path.join(target_path, f"with_crop{'_retina' if args.rl else ''}")
check_dir(save_path)

val_data = IJBDataset(img_path, lst_path, align=True, to_tensor=False)
img_names = val_data.img_names

batch_size = args.batch_size

data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                         num_workers=args.num_workers, drop_last=False, collate_fn=lambda x: x)

for idx, data in enumerate(tqdm(data_loader)):
    for sidx, (img, _) in enumerate(data):
        img_idx = idx * batch_size + sidx
        img_name = os.path.join(save_path, img_names[img_idx])
        cv2.imwrite(img_name, img[..., ::-1])
