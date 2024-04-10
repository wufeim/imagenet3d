# ImageNet3D

**ImageNet3D** dataset and helper code, from the following paper:

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

## Visualizing Sample Data

```sh
python3 visualize_sample_data.py
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
bbox = annot[0]['bbox']  # numpy.ndarray
category = annot[0]['class']  # str
principal_x = annot[0]['px']  # float
principal_y = annot[0]['py']  # float

# label indicating the quality of the object, occluded or low quality
object_status = annot[0]['object_status']  # str, one of ('status_good', 'status_partially', 'status_barely', 'status_bad')

# label indicating if multiple objects from same category very close to each other
dense = annot[0]['dense']  # str, one of ('dense_yes', 'dense_no')
```

## Citation
