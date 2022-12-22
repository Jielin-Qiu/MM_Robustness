# Image Perturbation

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
python perturb_Flickr30K_IP.py  

python perturbd_COCO_IP.py 

python perturb_NLVR_IP.py 

python perturb_VE_IP.py 
```

