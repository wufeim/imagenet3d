import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from render_utils import render


np.random.seed(42)
data_path = '/path/to/imagenet3d_data'


def visualize(img_path, annot_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    annot = dict(np.load(annot_path, allow_pickle=True))['annotations']

    for idx in range(len(annot)):
        if 'px' not in annot[idx]:
            continue

        px = annot[idx]['px']
        py = annot[idx]['py']
        viewport = annot[idx]['viewport']
        azimuth = annot[idx]['azimuth']
        elevation = annot[idx]['elevation']
        theta = annot[idx]['theta']
        distance = annot[idx]['distance']

        x1, y1, x2, y2 = annot[idx]['bbox']

        cad_index = int(annot[idx]['cad_index'])
        cad_path = os.path.join(data_path, 'CAD', annot[idx]['class'], f'{cad_index:02d}.off')

        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        img = render(img, cad_path, azimuth, elevation, theta, distance, px, py, viewport, good=(annot[idx]['ignore'] == 0))

    return img


save_path = 'examples'
n_samples = 10
for cate in sorted(os.listdir(os.path.join(data_path, 'annotations'))):
    annotation_files = os.listdir(os.path.join(data_path, 'annotations', cate))
    np.random.shuffle(annotation_files)
    annotation_files = annotation_files[:n_samples]

    os.makedirs(os.path.join(save_path, cate), exist_ok=True)

    for annot_file in tqdm(annotation_files, desc=cate):
        img_file = annot_file.replace('npz', 'JPEG')
        Image.fromarray(visualize(
            os.path.join(data_path, 'images', cate, img_file),
            os.path.join(data_path, 'annotations', cate, annot_file)
        )).save(os.path.join(save_path, cate, img_file))
