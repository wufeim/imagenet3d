import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from render_utils import render


np.random.seed(42)
data_path = '/path/to/imagenet3d_data'
processed_data_path = '/path/to/imagenet3d_data_processed'


def visualize(img_path, annot_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    annot = dict(np.load(annot_path, allow_pickle=True))

    px = annot['px']
    py = annot['py']
    viewport = annot['viewport']
    azimuth = annot['azimuth']
    elevation = annot['elevation']
    theta = annot['theta']
    distance = annot['distance']

    x1, y1, x2, y2 = annot['bbox']

    cad_index = int(annot['cad_index'])
    cad_path = os.path.join(data_path, 'CAD', str(annot['class']), f'{cad_index:02d}.off')

    img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
    img = render(img, cad_path, azimuth, elevation, theta, distance, px, py, viewport, good=True)

    return img


cates = sorted(os.listdir(os.path.join(processed_data_path, 'annotations')))
cates = ['_'.join(x.split('_')[:-1]) for x in cates]
cates = sorted(list(set(cates)))


save_path = 'examples_processed'
n_samples = 10
for cate in cates:
    annotation_files = os.listdir(os.path.join(processed_data_path, 'annotations', f'{cate}_train'))
    np.random.shuffle(annotation_files)
    annotation_files = annotation_files[:n_samples]

    os.makedirs(os.path.join(save_path, cate), exist_ok=True)

    for annot_file in tqdm(annotation_files, desc=cate):
        img_file = annot_file.replace('npz', 'JPEG')
        Image.fromarray(visualize(
            os.path.join(processed_data_path, 'images', f'{cate}_train', img_file),
            os.path.join(processed_data_path, 'annotations', f'{cate}_train', annot_file)
        )).save(os.path.join(save_path, cate, img_file))
