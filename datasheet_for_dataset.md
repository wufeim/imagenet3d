# Datasheet for dataset -- ImageNet3D

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

### For what purpose was the dataset created?

A vision model with general-purpose object-level 3D understanding should be capable of inferring both 2D (e.g., class name and bounding box) and 3D information (e.g., 3D location and 3D viewpoint) for arbitrary rigid objects in natural images. This is a challenging task, as it involves inferring 3D information from 2D signals and most importantly, generalizing to rigid objects from unseen categories. However, existing datasets with object-level 3D annotations are often limited by the number of categories or the quality of annotations. Models developed on these datasets become specialists for certain categories or domains, and fail to generalize. In this work, we present ImageNet3D, a large dataset for general-purpose object-level 3D understanding. ImageNet3D augments 200 categories from the ImageNet dataset with 2D bounding box, 3D pose, 3D location annotations, and image captions interleaved with 3D information. With the new annotations available in ImageNet3D, we could (i) analyze the object-level 3D awareness of visual foundation models, and (ii) study and develop general-purpose models that infer both 2D and 3D information for arbitrary rigid objects in natural images, and (iii) integrate unified 3D models with large language models for 3D-related reasoning.. We consider two new tasks, probing of object-level 3D awareness and open vocabulary pose estimation, besides standard classification and pose estimation. Experimental results on ImageNet3D demonstrate the potential of our dataset in building vision models with stronger general-purpose object-level 3D understanding.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

This dataset is created by the [CCVL group](https://ccvl.jhu.edu) at Johns Hopkins University. The development of the dataset is led by [Wufei Ma](https://wufeim.github.io), with help from [Guanning Zeng](https://scholar.google.com/citations?user=SU6ooAQAAAAJ), [Guofeng Zhang](https://openreview.net/profile?id=~Guofeng_Zhang4), and [Qihao Liu](https://qihao067.github.io/) on data collection and baseline experiments.

### Who funded the creation of the dataset?

*TBD.*

## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

Each object in ImageNet3D are annotated with the following:
- 3D viewpoint: parameterized by azimuth, elevation, and in-plane rotation
- 3D location: parameterized by 2D location and distance
- object visual quality
- scene density

Moreover, we provide category-level and image-level captions.

### How many instances are there in total (of each type, if appropriate)?

Over 86 thousand object instances.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

Our images and objects are obtained from ImageNet21k, which is a large dataset with diverse natural images. However, there are many more object instances varying by appearance or shape in the wild.

### What data does each instance consist of?

Images are a subset of ImageNet21k. We provide bounding boxes to each object instance in the image.

### Is there a label or target associated with each instance?

Yes. Annotations include 6D poses, object visual quality, scene density, and category-level and image-level captions.

### Is any information missing from individual instances?

N/A.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

N/A.

### Are there recommended data splits (e.g., training, development/validation, testing)?

Yes. We provide training and validation splits for various tasks. We also provide a list of "known" categories for open-vocabulary tasks.

### Are there any errors, sources of noise, or redundancies in the dataset?

N/A.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

Our dataset is self-contained.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

All our raw images are taken directly from ImageNet21k. Please follow the guidelines of ImageNet21k.

### Does the dataset relate to people?

No.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

No.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

Images may contain faces to identify individuals. However, all our raw images are taken directly from ImageNet21k so please follow the guidelines of ImageNet21k.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

No.

## Collection process

### How was the data associated with each instance acquired?

We recruit annotators to manually annotate these labels. Image captions are generated by GPT-4v.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

We developed a Python-based web app to host the annotation UI.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

N/A.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

Undergraduate students. They are paid at an hourly wage as specified by the institution.

### Over what timeframe was the data collected?

The timeframe spans from September 2023 to May 2024.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

Yes we obtained IRB approval from Johns Hopkins University prior to the start of our data annotation.

### Does the dataset relate to people?

No.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

Annotations are directly collected from the inputs of annotators.

### Were the individuals in question notified about the data collection?

Yes. Annotators are notified the purpose of the study and how the data may be used in research projects.

### Did the individuals in question consent to the collection and use of their data?

Yes.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

No. Annotators only provide 3D annotations on ImageNet images. We do not collect any images, videos, or personal data from the annotators.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

N/A.

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

No preprocessing or cleaning was applied to the raw data.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

Besides the raw data, we provide preprocessing code in our [GitHub repo](https://github.com/wufeim/imagenet3d) to produce object-centric images and remove images of low quality.

### Is the software used to preprocess/clean/label the instances available?

Our Python source code for preparing object-centric images is available in our [GitHub repo](https://github.com/wufeim/imagenet3d).

## Uses

### Has the dataset been used for any tasks already?

This dataset has been used for (i) linear probing of object-level 3D awareness, (ii) open-vocabulary pose estimation, and (iii) joint image classification and category-level pose estimation. Please refer to our [paper (coming soon...)](#) for details.

### Is there a repository that links to any or all papers or systems that use the dataset?

Currently no. (In the future we will host a benchmark on [paperswithcode.com](https://paperswithcode.com) to track models and papers using our dataset.)

### What (other) tasks could the dataset be used for?

Our ImageNet3D dataset can be used in many other 2D and 3D computer vision tasks, development general-purpose 3D models or evaluating 3D-awarness of vision models.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

As our image data are collected from ImageNet21k, most images are object-centric with only one or two instances. Thus our dataset may not be suitable for 3D object detection or tasks that require object co-occurrences.

### Are there tasks for which the dataset should not be used?

N/A.

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?

N/A.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

The raw data is hosted on [HuggingFace](https://huggingface.co/datasets/ccvl/ImageNet3D). Please refer to our [GitHub repo](https://github.com/wufeim/imagenet3d) for instructions, sample usage, and baseline experiments. The dataset is also accessible from [doi.org/10.57967/hf/2461](https://doi.org/10.57967/hf/2461).

### When will the dataset be distributed?

An early version of our dataset was first available on Apr 09, 2024.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

N/A.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Our dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) license. Additionally users must abid to the license and terms of accesss of the original [ImageNet dataset](https://www.image-net.org/download.php)

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

N/A.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

Our group at [CCVL](https://ccvl.jhu.edu) will be supporting and maintaining the dataset in the future. This includes fixing possible issues with the dataset or further polishing the annotation quality.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

Please contact [Wufei Ma](https://wufeim.github.io) regarding any issues or concerns with the ImageNet3D dataset.

### Is there an erratum?

No.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

We will post any future updates on our [GitHub repo](https://github.com/wufeim/imagenet3d) and our [project page](https://wufeim.github.io/imagenet3d/index.html).

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

N/A.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Older versions of our dataset will continue to be available for reference.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Others are encouraged to extend/augment/build on/contribute to our dataset, as long as they follow the license and terms of access of both our ImageNet3D and the original [ImageNet dataset](https://www.image-net.org).

We also welcome collaboration to improve and extend the ImageNet3D dataset. If you are interested, please reach out to [Wufei Ma](https://wufeim.github.io).
