import argparse
import multiprocessing
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ImageNet3D data')
    # parser.add_argument('--data_path', type=str, default='/path/to/imagenet3d')
    # parser.add_argument('--save_path', type=str, default='/path/to/imagenet3d_processed')
    parser.add_argument('--data_path', type=str, default='/home/wufeim/research/imagenet3d_data/imagenet3d_data')
    parser.add_argument('--save_path', type=str, default='/home/wufeim/research/imagenet3d_data/imagenet3d_data_processed')
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    parser.add_argument('--categories', type=str, nargs='+', default=None)

    parser.add_argument('--image_height', type=int, default=640)
    parser.add_argument('--image_width', type=int, default=800)
    parser.add_argument('--center_and_resize', action='store_true')
    parser.add_argument('--target_distance', type=float, default=4.0)
    return parser.parse_args()


def preprocess(cate, sample, img_path, annot_path, save_img_path, save_annot_path, args):
    annot = dict(np.load(annot_path, allow_pickle=True))['annotations']

    preprocessed_samples = []
    for obj_id in range(len(annot)):
        if annot[obj_id]['ignore'] > 0:
            continue

        if annot[obj_id]['distance'] <= 0.0:
            continue

        img = np.array(Image.open(img_path).convert('RGB'))

        raw_dist = annot[obj_id]['distance']
        if args.center_and_resize:
            resize_rate = raw_dist / args.target_distance
        else:
            resize_rate = min(args.image_height / img.shape[0], args.image_width / img.shape[1])
        new_dist = raw_dist / resize_rate

        dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
        img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)

        bbox_raw = annot[obj_id]['bbox']
        bbox_new = bbox_raw * resize_rate

        px_raw, py_raw = annot[obj_id]['px'], annot[obj_id]['py']
        if args.center_and_resize:
            center_x, center_y = int(px_raw * resize_rate), int(py_raw * resize_rate)
            px_new, py_new = args.image_width / 2, args.image_height / 2
        else:
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            px_new = px_raw * resize_rate
            py_new = py_raw * resize_rate

        padding = (
            (
                max(args.image_height // 2 - center_y, 0),
                max(args.image_height // 2 + center_y - img.shape[0], 0)
            ),
            (
                max(args.image_width // 2 - center_x, 0),
                max(args.image_width // 2 + center_x - img.shape[1], 0)
            ),
            (0, 0)
        )
        img = np.pad(img, padding, mode='constant')
        bbox_new[0] += padding[1][0]
        bbox_new[2] += padding[1][0]
        bbox_new[1] += padding[0][0]
        bbox_new[3] += padding[0][1]
        if not args.center_and_resize:
            px_new += padding[1][0]
            py_new += padding[0][0]

        cropping = (
            (
                max(center_y - args.image_height // 2, 0),
                max(img.shape[0] - center_y - args.image_height // 2, 0)
            ),
            (
                max(center_x - args.image_width // 2, 0),
                max(img.shape[1] - center_x - args.image_width // 2, 0)
            ),
            (0, 0)
        )
        img = img[cropping[0][0]:cropping[0][0] + args.image_height, cropping[1][0]:cropping[1][0] + args.image_width, :]
        bbox_new[0] -= cropping[1][0]
        bbox_new[2] -= cropping[1][0]
        bbox_new[1] -= cropping[0][0]
        bbox_new[3] -= cropping[0][0]
        if not args.center_and_resize:
            px_new -= cropping[1][0]
            py_new -= cropping[0][0]

        name = f'{sample}_{obj_id:02d}'
        save_parameters = dict(
            name=name,
            bbox=bbox_new,
            bbox_original=bbox_raw,
            principal=np.array([px_new, py_new]),
            principal_original=annot[obj_id]['principal'],
            px=px_new,
            py=py_new,
            px_original=annot[obj_id]['px'],
            py_original=annot[obj_id]['py'],
            distance=new_dist,
            distance_original=raw_dist,
            azimuth=annot[obj_id]['azimuth'],
            elevation=annot[obj_id]['elevation'],
            theta=annot[obj_id]['theta'],
            cad_index=annot[obj_id]['cad_index'],
            focal=annot[obj_id]['focal'],
            viewport=annot[obj_id]['viewport'],
            padding_params=np.array([padding[0][0], padding[0][1], padding[1][0], padding[1][1], padding[2][0], padding[2][1]]),
            cropping_params=np.array([cropping[0][0], cropping[0][1], cropping[1][0], cropping[1][1], cropping[2][0], cropping[2][1]]),
            resize_rate=resize_rate)
        save_parameters['class'] = cate

        np.savez(os.path.join(save_annot_path, name), **save_parameters)
        Image.fromarray(img).save(os.path.join(save_img_path, name+'.JPEG'))
        preprocessed_samples.append(name)

    return preprocessed_samples, {'total': len(annot), 'processed': len(preprocessed_samples), 'skipped': len(annot)-len(preprocessed_samples)}


def worker(params):
    args, split, cate = params

    raw_img_path = os.path.join(args.data_path, 'images', cate)
    raw_annot_path = os.path.join(args.data_path, 'annotations', cate)

    with open(os.path.join(args.data_path, 'lists', f'{cate}_imagenet_{split}.txt'), 'r') as fp:
        all_samples = fp.read().strip().split('\n')

    save_img_path = os.path.join(args.save_path, 'images', f'{cate}_{split}')
    save_annot_path = os.path.join(args.save_path, 'annotations', f'{cate}_{split}')
    save_list_path = os.path.join(args.save_path, 'lists', f'{cate}_{split}')
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_annot_path, exist_ok=True)
    os.makedirs(save_list_path, exist_ok=True)

    all_preprocessed_samples = []
    stats = {'total': 0, 'processed': 0, 'skipped': 0}
    for sample in all_samples:
        img_path = os.path.join(raw_img_path, f'{sample}.JPEG')
        annot_path = os.path.join(raw_annot_path, f'{sample}.npz')

        preprocessed_samples, _stats = preprocess(
            cate,
            sample,
            img_path,
            annot_path,
            save_img_path,
            save_annot_path,
            args)

        all_preprocessed_samples += preprocessed_samples
        for k in stats:
            stats[k] += _stats[k]

    with open(os.path.join(save_list_path, 'mesh01.txt'), 'w') as fp:
        fp.write('\n'.join(all_preprocessed_samples))

    return {'cate': cate, 'split': split, 'stats': stats}


def main():
    args = parse_args()

    if args.categories is None:
        args.categories = os.listdir(os.path.join(args.data_path, 'images'))
    args.categories = sorted(args.categories)

    tasks = []
    for cate in args.categories:
        for split in args.splits:
            tasks.append([args, split, cate])

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    all_stats = {'total': 0, 'processed': 0, 'skipped': 0}
    results_dict = {}
    for r in results:
        for k in all_stats:
            all_stats[k] += r['stats'][k]
        if r['cate'] not in results_dict:
            results_dict[r['cate']] = {}
        results_dict[r['cate']][r['split']] = r['stats']

    print('Preprocessing finished')
    print(f'\tTotal objects: {all_stats["total"]}')
    print(f'\tProcessed objects: {all_stats["processed"]}')
    print(f'\tSkipped objects: {all_stats["skipped"]}')

    for c in args.categories:
        for s in args.splits:
            stats = results_dict[c][s]
            print(f'{c} {s}: t={stats["total"]} p={stats["processed"]} s={stats["skipped"]}')


if __name__ == '__main__':
    main()
