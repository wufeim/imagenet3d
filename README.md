# ImageNet3D

**ImageNet3D** dataset and helper code, from the following paper:

ImageNet3D: Towards General-Purpose Object-Level 3D Understanding. Preprint, 2024.\
[Wufei Ma](https://wufeim.github.io), [Guanning Zeng](https://scholar.google.com/citations?user=SU6ooAQAAAAJ), [Qihao Liu](https://qihao067.github.io/), [Letian Zhang](https://scholar.google.com/citations?hl=en&user=o25Si3QAAAAJ), [Adam Kortylewski](https://adamkortylewski.com/), [Yaoyao Liu](https://www.cs.jhu.edu/~yyliu/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)\
Johns Hopkins University\
[[arXiv]](https://arxiv.org/abs/2406.09613), [[Project Page]](https://wufeim.github.io/imagenet3d/index.html)

## Overview

All available CAD models are available [here](vis_models.md).

For details of ImageNet3D, please refer to [datasheet for dataset](datasheet_for_dataset.md).

## Installation

## Download ImageNet3D

Modify the `local_dir` parameter to your local directory.

```py
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ccvl/imagenet3d-0409',
    repo_type='dataset',
    filename='imagenet3d_0409.zip',
    local_dir='/path/to/imagenet3d_0409.zip',
    local_dir_use_symlinks=False)
```

## Usage

Visualizing sample raw data.

```sh
python3 visualize_sample_data.py
```

Preprocessing data for 3D pose estimation.

```sh
python3 preprocess_data.py --center_and_resize
```

Preprocessing data for 6D pose estimation.

```sh
python3 preprocess_data.py
```

Visualizing sample preprocessed data.

```sh
python3 visualize_sample_data_processed.py
```

## Annotation Example

```py
from PIL import Image
import numpy as np

img_path = 'imagenet3d/bed/n02818832_13.JPEG'
annot_path = 'imagenet3d/bed/n02818832_13.npz'

img = np.array(Image.open(img_path).convert('RGB'))
annot = dict(np.load(annot_path, allow_pickle=True))['annotations']

# Number of objects
num_objects = len(annot)

# Annotation of the first object
azimuth = annot[0]['azimuth']  # float, [0, 2*pi]
elevation = annot[0]['elevation']  # float, [0, 2*pi]
theta = annot[0]['theta']  # float, [0, 2*pi]
cad_index = annot[0]['cad_index']  # int
distance = annot[0]['distance']  # float
viewport = annot[0]['viewport']  # int
img_height = annot[0]['height']  # numpy.uint16
img_width = annot[0]['width']  # numpy.uint16
bbox = annot[0]['bbox']  # numpy.ndarray, (x1, y1, x2, y2)
category = annot[0]['class']  # str
principal_x = annot[0]['px']  # float
principal_y = annot[0]['py']  # float

# label indicating the quality of the object, occluded or low quality
object_status = annot[0]['object_status']  # str, one of ('status_good', 'status_partially', 'status_barely', 'status_bad')

# label indicating if multiple objects from same category very close to each other
dense = annot[0]['dense']  # str, one of ('dense_yes', 'dense_no')
```

* `object status`: quality of the object:
  * Good (`status_good`): most parts of the object is visible in the image
  * Partially visible (`status_partially`): a small part of the object is occluded by other objects or outside the image
  * Barely visible (`status_barely`): only a small part of the object is visible; the other parts are occluded or outside the image
  * Bad quality / no object (`status_bad`): most parts of the object is occluded or outside the image; we can see there is an object but very hard to tell the pose of the object
* `dense`: if the object is very close to another object from the same category; here “close” is defined in the 2D image plane – two objects are close if the distance between them is small in the 2D image plane
  * Not dense scene (`dense_no`): the object is not close to another object from the same category; there can be multiple objects from the same category in one image but the objects are far away from each other
  * Dense scene (`dense_yes`): the object is very close to another object from the same category; they may occlude each other or just very close – imagine a parking lot where cars are close to each other.

## Citation

```
@article{ma2024imagenet3d,
  title={ImageNet3D: Towards General-Purpose Object-Level 3D Understanding},
  author={Ma, Wufei and Zeng, Guanning and Zhang, Guofeng and Liu, Qihao and Zhang, Letian and Kortylewski, Adam and Yuille, Alan},
  journal={arXiv preprint	arXiv:2406.09613},
  year={2024}
}
```
