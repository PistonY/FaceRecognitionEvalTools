import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model, check_dir, l2norm, get_model_info
from IJB.utils import read_meta_file, generate_image_feature

parser = argparse.ArgumentParser(description='do ijb test.')
parser.add_argument('--model-path', type=str, required=True,
                    help='only torchscript model is accept.')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--target', default='IJBC', type=str,
                    help='target, set to IJBC or IJBB')
parser.add_argument('--root-path', required=True, type=str,
                    help='Data root path.')
parser.add_argument('--use-aligned-image', action='store_true',
                    help='Use cropped img or not.')
parser.add_argument('--retina-landmark', dest='rl', action='store_true',
                    help='use retina face landmark.')
parser.add_argument('--flip', action='store_true',
                    help='use flip added')
parser.add_argument('--det-score', action='store_true',
                    help='use detector score')
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--save-path', type=str, required=True)
args = parser.parse_args()


def image_template_feature(img_feats, template, media):
    template = np.array(template, np.int)
    media = np.array(media, np.int)
    unique_template = np.unique(template)
    template_norm_feats = []
    for uqt in tqdm(unique_template):
        ind_t = np.where(template == uqt)
        face_feats = img_feats[ind_t]
        face_media = media[ind_t]
        unique_media, unique_media_counts = np.unique(face_media, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_media, unique_media_counts):
            ind_m = np.where(face_media == u)
            if ct == 1:
                media_norm_feats.append(face_feats[ind_m])
            else:
                media_norm_feats.append(np.mean(face_feats[ind_m], axis=0, keepdims=True))
        media_norm_feats = np.array(media_norm_feats)
        template_norm_feats.append(np.sum(media_norm_feats, axis=0))

    template_norm_feats = np.concatenate(template_norm_feats, axis=0)
    template_norm_feats = l2norm(template_norm_feats)
    return template_norm_feats, unique_template


def verification(template_norm_feats, unique_template, p1, p2):
    p1, p2 = np.array(p1, np.int), np.array(p2, np.int)
    template2id = np.zeros((max(unique_template) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_template):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    for s in tqdm(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
    return score


if __name__ == '__main__':
    assert args.target in ('IJBC', 'IJBB')

    use_flip_test = args.flip  # if True, TestMode(F1)
    use_detector_score = args.det_score  # if True, TestMode(D1)
    use_aligned_img = args.use_aligned_image

    print(f'Tlip test is enabled: {use_flip_test}, \n'
          f'Use detector score is enabled: {use_detector_score}.\n')

    IJB_root = os.path.join(args.root_path, args.target)
    # =============================================================
    # load image and template relationships for template feature embedding
    # tid --> template id,  mid --> media id
    # format:
    #           image_name tid mid
    # =============================================================
    print('Loading template id and media id.')
    _, templates, medias = read_meta_file(
        os.path.join(IJB_root, 'meta', '{}_face_tid_mid.txt'.format(args.target.lower())),
        dtypes=(str, int, int))
    print('Load done.\n')

    # =============================================================
    # load template pairs for template-to-template verification
    # tid : template id,  label : 1/0
    # format:
    #           tid_1 tid_2 label
    # =============================================================
    print('Loading template pairs for template-to-template verification.')
    p1, p2, label = read_meta_file(
        os.path.join(IJB_root, 'meta', '{}_template_pair_label.txt'.format(args.target.lower())),
        dtypes=(int, int, int))
    lable_path = os.path.join(IJB_root, 'meta', 'label.npy')
    if not os.path.exists(lable_path):
        np.save(lable_path, np.array(label, np.int))
    print('Load done.\n')

    # =============================================================
    # load model
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, embedding_size)
    # =============================================================
    print('Loading model and generate features.\n')

    model = load_model(args.model_path, device='cuda:0', mode='script', parallel=True)
    embedding_size, dtype = get_model_info(model, (1, 3, 112, 112), device='cuda:0')
    print(f'Model Info: embedding size: {embedding_size}, datatype: {dtype}.')

    if not use_aligned_img:
        img_folder = 'loose_crop'
    elif not args.rl:
        img_folder = 'with_crop'
    else:
        img_folder = 'with_crop_retina'
    img_root = os.path.join(IJB_root, img_folder)
    img_list_path = os.path.join(IJB_root, 'meta', '{}_name_5pts_score.txt'.format(args.target.lower()))
    img_feats = generate_image_feature(img_list_path, img_root, model, args.batch_size, args.num_workers,
                                       dtype, use_flip_test, use_detector_score, use_aligned_img)
    print(f'Done! Feature Shape: ({img_feats.shape[0]} , {img_feats.shape[1]}) .\n')

    # =============================================================
    # compute template features from image features.
    # =============================================================
    print('Computing template features from image features.')
    template_feats, unique_templates = image_template_feature(img_feats, templates, medias)
    print('Done!\n')

    # =============================================================
    # compute verification scores between template pairs.
    # =============================================================
    print('Computing verification scores between template pairs and save.')
    score = verification(template_feats, unique_templates, p1, p2)
    save_path = os.path.join(args.save_path, args.target)
    check_dir(save_path)
    model_name = args.model_path.split('/')[-1].split('.')[0]
    np.save(os.path.join(save_path, f"{model_name}"
                                    f"{'F1' if use_flip_test else 'F0'}"
                                    f"{'D1' if use_detector_score else 'D0'}"
                                    f"{'Retina' if args.rl else ''}.npy"), score)
    print('Finish verification and Saving.')
