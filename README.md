# MM_Robustness_Benchmark

The code for generating multimodal robustness evaluation datasets

## Installation

```
cd image_perturbation/imagenet_c/
```
```
pip install -e .
```
```
cd ..
```

## Download the Flickr30K and COCO datasets

The datasets can be downloaded from the original website:

Flickr30K: https://shannon.cs.illinois.edu/DenotationGraph/

COCO: https://cocodataset.org/#home

Or can be downloaded from google drive:

Flickr30K: 

COCO:


## Generate Flickr30K-IP dataset
```
cd Image_perturbation
```
Download the image_perturbation/flickr30k_images_test.tar.gz file which contains the 1000 Flickr30K test images.
```
tar -zxvf flickr30k_images_test.tar.gz
```
```
python perturbated_data_Flickr30K.py to generate all the perturbated data for 5 severity levels and 15 pertubation strategies. 
```

## Generate COCO-IP dataset

Download the image_perturbation/coco_images_test.tar.gz file which contains the 5000 COCO test images.
```
tar -zxvf coco_images_test.tar.gz
```
```
python perturbated_data_COCO.py to generate all the perturbated data for 5 severity levels and 15 pertubation strategies. 
```

## Generate Flickr30K-TP dataset
```
cd text_perturbation/eda_nlp
```
```
python run_text_aug_f30k_v2.py
```
The script will produce 5 level of each perturbation method.

The generated files can also be found under './f30k_eda'.

## Generate COCO-TP dataset
```
cd text_perturbation/eda_nlp
```
```
python run_text_aug_coco_v2.py
```
The script will produce 5 level of each perturbation method. 

The generated files can also be found under './coco_eda'.
