# MM_Robustness

Journal of Data-centric Machine Learning Research (DMLR)

## [Benchmarking Robustness of Multimodal Image-Text Models under Distribution Shift](https://arxiv.org/abs/2212.08044)

More details can be found on the [project webpage](https://mmrobustness.github.io/).

The code for generating multimodal robustness evaluation datasets for downstream image-text applications, including image-text retrieval, visual reasoning, visual entailment, image captioning, and text-to-image generation.

## Citation

If you feel our code or models help your research, kindly cite our papers:

```
@inproceedings{Qiu2022BenchmarkingRO,
  title={Benchmarking Robustness of Multimodal Image-Text Models under Distribution Shift},
  author={Jielin Qiu and Yi Zhu and Xingjian Shi and F. Wenzel and Zhiqiang Tang and Ding Zhao and Bo Li and Mu Li},
  journal={Journal of Data-centric Machine Learning Research (DMLR)},
  year={2024}
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

For the text-to-image generation evaluation, we used the captions from COCO as prompt to generate the corresponding images. We also share the generated images [here](https://drive.google.com/drive/folders/1V8ejnA0y59wchKfsMFU8Y9XIOPPZQNiN?usp=sharing).

## Baselines

For the evaluated baselines, plase see [evaluated_baselines](https://github.com/Jason-Qiu/MM_Robustness/tree/main/evaluated_baselines)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.


## License

This project is licensed under the Apache-2.0 License.
