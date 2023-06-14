# MM_Robustness

## [Are Multimodal Models Robust to Image and Text Perturbations?](https://arxiv.org/abs/2212.08044)

More details can be found on the [project webpage](https://mmrobustness.github.io/).

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

## Generate perturbation datasets

- For image perturbation, please see [image_perturbation](https://github.com/Jason-Qiu/MM_Robustness/tree/main/image_perturbation)

- For text perturbation, please see [text_perturbation](https://github.com/Jason-Qiu/MM_Robustness/tree/main/text_perturbation)

- For detection score, please see [detection_score](https://github.com/Jason-Qiu/MM_Robustness/tree/main/detection_score)

## Evaluation data for text-to-image generation

For the text-to-image generation evaluation, we used the captions from COCO as prompt to generate the corrsponding images. We also share the generated images [here](https://drive.google.com/drive/folders/1V8ejnA0y59wchKfsMFU8Y9XIOPPZQNiN?usp=sharing).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.


## License

This project is licensed under the Apache-2.0 License.
