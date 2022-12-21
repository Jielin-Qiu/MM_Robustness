# MM_Robustness

The code for generating multimodal robustness evaluation datasets for downstream image-text applications, including image-text retrieval, visual reseaning, and visual entailment.

## Installation

```
./install.sh
```

## Datasets

- The original datasets can be downloaded from the original website:
  - Flickr30K: https://shannon.cs.illinois.edu/DenotationGraph/
  - COCO: https://cocodataset.org/#home
  - NLVR2: https://lil.nlp.cornell.edu/nlvr/
  - SNLI-VE: https://github.com/necla-ml/SNLI-VE

- The unperturbed test/val image data can also be downloaded from google drive: 

  - Flickr30K: [test](https://drive.google.com/file/d/1UfoHywRWYgiE6NHh398yMQTzqKllvIZR/view?usp=sharing)
  - COCO: [test](https://drive.google.com/file/d/1zPA3yiB3sXXdjLUV0bPkGqOX840MXoGH/view?usp=sharing)
  - NLVR2: [dev](https://drive.google.com/file/d/10qRZP65Lhkww_Be5XLLM2AHsntgglwLN/view?usp=sharing), [test1](https://drive.google.com/file/d/1RhXAumgH_QGZa29BWqcqC19-cKpJf9fm/view?usp=sharing)
  - SNLI-VE: [val](https://drive.google.com/file/d/14l1XdsFnpJcY7OOixL0xUqERc5QLefnI/view?usp=sharing)，[test](https://drive.google.com/file/d/1NyXK-Vw1UDQiZ-APqE5C92XI6Ip_HWMW/view?usp=sharing)

- The unperturbed test/val text data can be found under `./original_annotation` 
- The unperturbed test/val text data can also be downloaded from google drive: [TP-annotations](https://drive.google.com/file/d/1rTmYOasXACXm1PGcptOEDmuKqE6pQiPW/view?usp=sharing)


## Generate image perturbation (IP) dataset

- The stylize perturbation is based on [stylize-datasets](https://github.com/bethgelab/stylize-datasets) and [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet). You will need to download the models (vgg/decoder) manually from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and move both files to the `stylize-datasets/models/` directory. You will need to download train.zip from [Kaggle's painter-by-numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data).

Due to the size of the perturbed data, we didn't provide the perturbed images, but they can be generated easily with the following command:

```
python perturb_Flickr30K_IP.py  
```
```
python perturbd_COCO_IP.py 
```
```
python perturb_NLVR_IP.py 
```
```
python perturb_VE_IP.py 
```

## Generate text perturbation (TP) dataset

```
python perturb_Flickr30K_TP.py  
```
```
python perturbd_COCO_TP.py 
```
```
python perturb_NLVR_TP.py 
```
```
python perturb_VE_TP.py 
```

## Citation

If you feel our code or models helps in your research, kindly cite our papers:

```
@inproceedings{Qiu2022AreMM,
  title={Are Multimodal Models Robust to Image and Text Perturbations?},
  author={Jielin Qiu and Yi Zhu and Xingjian Shi and Florian Wenzel and Zhiqiang Tang and Ding Zhao and Bo Li and Mu Li},
  journal={arXiv preprint arXiv:2212.08044},
  year={2022}
}
```

