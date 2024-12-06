# ImageNet3D

**ImageNet3D** dataset and helper code, from the following paper:

[ImageNet3D: Towards General-Purpose Object-Level 3D Understanding](https://arxiv.org/abs/2406.09613). NeurIPS, 2024.\
[Wufei Ma](https://wufeim.github.io), [Guofeng Zhang](https://openreview.net/profile?id=~Guofeng_Zhang4), [Qihao Liu](https://qihao067.github.io/), [Guanning Zeng](https://scholar.google.com/citations?user=SU6ooAQAAAAJ), [Adam Kortylewski](https://adamkortylewski.com/), [Yaoyao Liu](https://www.cs.jhu.edu/~yyliu/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)\
Johns Hopkins University\
[[`arXiv`]](https://arxiv.org/abs/2406.09613) [[`Project Page`]](https://wufeim.github.io/imagenet3d/index.html) [[`NeurIPS website`]](https://nips.cc/virtual/2024/poster/97507)

## Overview

All available CAD models are available [here](vis_models.md).

For details of ImageNet3D, please refer to [datasheet for dataset](datasheet_for_dataset.md).

## Download Data

**ImageNet3D-v1.0:** Directly download from the HuggingFace WebUI, or on a server, run

```sh
wget https://huggingface.co/datasets/ccvl/ImageNet3D/resolve/main/imagenet3d_v1.zip
```

**Future updates:** We are working on annotating more object categories and improving the quality of current annotations. The next update is planned to be released by the end of Jan 2025. Please let us know if you have any suggestions for future updates.

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
  author={Ma, Wufei and Zhang, Guofeng and Liu, Qihao and Zeng, Guanning and Kortylewski, Adam and Liu, Yaoyao and Yuille, Alan},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2024}
}
```
