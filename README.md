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

## Download the datasets

The datasets can be downloaded from the original website:

Flickr30K: https://shannon.cs.illinois.edu/DenotationGraph/

COCO: https://cocodataset.org/#home

NLVR2: https://lil.nlp.cornell.edu/nlvr/

SNLI-VE: https://github.com/necla-ml/SNLI-VE

Or the test data can be downloaded from google drive:

Flcikr30K: [test](https://drive.google.com/file/d/1UfoHywRWYgiE6NHh398yMQTzqKllvIZR/view?usp=sharing)

COCO: [test](https://drive.google.com/file/d/1zPA3yiB3sXXdjLUV0bPkGqOX840MXoGH/view?usp=sharing)

NLVR2: [dev](https://drive.google.com/file/d/10qRZP65Lhkww_Be5XLLM2AHsntgglwLN/view?usp=sharing), [test1](https://drive.google.com/file/d/1RhXAumgH_QGZa29BWqcqC19-cKpJf9fm/view?usp=sharing)

SNLI-VE: [val](https://drive.google.com/file/d/14l1XdsFnpJcY7OOixL0xUqERc5QLefnI/view?usp=sharing)，[test](https://drive.google.com/file/d/1NyXK-Vw1UDQiZ-APqE5C92XI6Ip_HWMW/view?usp=sharing)


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
