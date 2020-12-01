import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import math
import heapq
import numpy as np
from tqdm import tqdm
from Eval.utils import load_model, l2norm, get_model_info
from Eval.IJB.utils import read_meta_file, generate_image_feature

parser = argparse.ArgumentParser(description='do ijb test.')
parser.add_argument('--model-path', type=str, required=True,
                    help='only torchscript model is accept.')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--root-path', required=True, type=str, help='Data root path.')
parser.add_argument('--use-aligned-image', action='store_true', help='Use cropped img or not.')
parser.add_argument('--retina-landmark', dest='rl', action='store_true',
                    help='use retina face landmark.')
parser.add_argument('--flip', action='store_true', help='use flip added')
parser.add_argument('--det-score', action='store_true', help='use detector score')
parser.add_argument('--num-workers', type=int, default=0)

args = parser.parse_args()


def gen_mask(query_ids, reg_ids):
    mask = []
    for query_id in query_ids:
        pos = [i for i, x in enumerate(reg_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError(
                "RegIdsError with id = {}， duplicate = {} ".format(
                    query_id, len(pos)))
        mask.append(pos[0])
    return mask


def read_template_subject_id_list(path):
    ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
    templates = ijb_meta[:, 0].astype(np.int)
    subject_ids = ijb_meta[:, 1].astype(np.int)
    return templates, subject_ids


def image_template_feature(img_feats, template, media, choose_templates, choose_ids):
    template = np.array(template, np.int)
    media = np.array(media, np.int)
    unique_templates, indices = np.unique(choose_templates, return_index=True)
    unique_subjectids = choose_ids[indices]
    template_feats = []
    for uqt in tqdm(unique_templates):
        ind_t = np.where(template == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = media[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            ind_m = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats.append(face_norm_feats[ind_m])
            else:
                media_norm_feats.append(np.mean(face_norm_feats[ind_m], axis=0, keepdims=True))
        media_norm_feats = np.array(media_norm_feats)
        template_feats.append(np.sum(media_norm_feats, axis=0))
    template_feats = np.concatenate(template_feats, axis=0)
    template_norm_feats = l2norm(template_feats)

    return template_norm_feats, unique_templates, unique_subjectids


def evaluation(query_feats, gallery_feats, mask):
    Fars = [0.01, 0.1]

    query_num = query_feats.shape[0]
    # gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num / query_num))
    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num / query_num))
    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:10]
        if mask[i] in j:
            correct_num += 1
    print("top10 = {}".format(correct_num / query_num))

    # neg_pair_num = query_num * gallery_num - query_num

    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)

    neg_sims = top_sims[np.where(top_sims > -2.0)]
    print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))


if __name__ == '__main__':
    target = args.target
    IJB_root = os.path.join(args.root_path, target)
    meta_dir = os.path.join(IJB_root, 'meta')
    use_aligned_img = args.use_aligned_image
    use_flip_test = args.flip
    use_detector_score = args.det_score

    if target == 'IJBC':
        gallery_s1_record = f'{target.lower()}_1N_gallery_G1.csv'
        gallery_s2_record = f'{target.lower()}_1N_gallery_G2.csv'
    else:
        gallery_s1_record = f'{target.lower()}_1N_gallery_S1.csv'
        gallery_s2_record = f'{target.lower()}_1N_gallery_S2.csv'

    print('Reading gallery stuff.')
    gallery_s1_templates, gallery_s1_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))

    gallery_s2_templates, gallery_s2_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))

    gallery_templates = np.concatenate([gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate([gallery_s1_subject_ids, gallery_s2_subject_ids])

    _, total_templates, total_medias = read_meta_file(
        os.path.join(IJB_root, 'meta', '{}_face_tid_mid.txt'.format(args.target.lower())),
        dtypes=(str, int, int))
    print('Done!')

    print('Loading model and generate features.')

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
    print('Done!')

    print('Computing template features from image features.')
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image_template_feature(
        img_feats, total_templates, total_medias, gallery_templates, gallery_subject_ids)

    probe_mixed_record = f'{target.lower()}_1N_probe_mixed.csv'
    probe_mixed_templates, probe_mixed_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))

    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image_template_feature(
        img_feats, total_templates, total_medias, probe_mixed_templates, probe_mixed_subject_ids)
    print('Done!')

    print('Start evaluation.')
    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = gen_mask(probe_ids, gallery_ids)

    evaluation(probe_feats, gallery_feats, mask)
