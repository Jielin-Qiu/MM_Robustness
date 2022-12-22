# MM_Robustness

The code for generating multimodal robustness evaluation datasets for downstream image-text applications, including image-text retrieval, visual reseaning, visual entailment, image captioning, and text-to-image generation.

## Citation

If you feel our code or models helps in your research, kindly cite our papers:

```
@inproceedings{Qiu2022AreMM,
  title={Are Multimodal Models Robust to Image and Text Perturbations?},
  author={Jielin Qiu and Yi Zhu and Xingjian Shi and Florian Wenzel and Zhiqiang Tang and Ding Zhao and Bo Li and Mu Li},
  journal={arXiv preprint arXiv:2212.08044},
  year={2022}
}

@inproceedings{qiu2022benchmarking,
  title={Benchmarking Robustness under Distribution Shift of Multimodal Image-Text Models},
  author={Qiu, Jielin and Zhu, Yi and Shi, Xingjian and Tang, Zhiqiang and Zhao, Ding and Li, Bo and Li, Mu},
  booktitle={NeurIPS 2022 Workshop on Distribution Shifts: Connecting Methods and Applications}
}
```

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

## Generate image perturbation (IP) dataset

- The stylize perturbation is based on [stylize-datasets](https://github.com/bethgelab/stylize-datasets) and [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet). You will need to download the models (vgg/decoder) manually from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and move both files to the `stylize-datasets/models/` directory. You will need to download train.zip from [Kaggle's painter-by-numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data).

Due to the size of the perturbed data, we didn't provide the perturbed images, but they can be generated easily with the following command:

```
cd image_perturbation

python perturb_Flickr30K_IP.py  

python perturbd_COCO_IP.py 

python perturb_NLVR_IP.py 

python perturb_VE_IP.py 
```

## Generate text perturbation (TP) dataset

```
cd text_perturbation

python perturb_Flickr30K_TP.py  

python perturbd_COCO_TP.py 

python perturb_NLVR_TP.py 

python perturb_VE_TP.py 
```


