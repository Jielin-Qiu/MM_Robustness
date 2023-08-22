## Evaluated Baselines

In our experiments, we evaluated 9 multimodal models. The models are publicly available in the following repos: 

CLIP: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
ViLT: [https://github.com/dandelin/ViLT](https://github.com/dandelin/ViLT)  
TCL: [https://github.com/uta-smile/TCL](https://github.com/uta-smile/TCL)  
ALBEF: [https://github.com/salesforce/ALBEF](https://github.com/salesforce/ALBEF)  
BLIP: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)  
METER: [https://github.com/zdou0830/METER](https://github.com/zdou0830/METER)  
GRIT: [https://github.com/JialianW/GRiT](https://github.com/JialianW/GRiT)  
Stable Diffusion: [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)  
GLIDE: [https://github.com/bumptech/glide](https://github.com/bumptech/glide)  

## Experimental hyperparameters

#### Image-text retrieval

- CLIP: Our results are based on [CLIP ViT-L/14@336px](https://huggingface.co/openai/clip-vit-large-patch14-336)
- ViLT: We used the weights [ViLT-B/32 200k finetuned on COCO IR/TR](https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_coco.ckpt), [ViLT-B/32 200k finetuned on F30K IR/TR](https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_f30k.ckpt) for COCO and Flickr, respectively. 
- TCL: We used the weights [TCL_4M](https://drive.google.com/file/d/1Cb1azBdcdbm0pRMFs-tupKxILTCXlB4O/view?usp=sharing), [TCL_Retrieval_Coco_Finetune](https://drive.google.com/file/d/1PtcZF_XzJgIceg4rXLWqGQiXjizvxxS6/view?usp=sharing), [TCL_Retrieval_Flickr_Finetune](https://drive.google.com/file/d/1qwWfqyCu1F5YZqQNxjkqy1REESoU6pOT/view?usp=sharing) for COCO and Flickr, respectively. 
- ALBEF: We used the weights [Finetuned checkpoint for retrieval on MSCOCO](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth), [Finetuned checkpoint for retrieval on Flickr30k](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth)  for COCO and Flickr, respectively. 
- BLIP: We used the weights [Image-Text Retrieval (COCO)](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth), [Image-Text Retrieval (Flickr30k)](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth)  for COCO and Flickr, respectively. 

#### Visual Reasoning

- ALBEF:  We follow the commands in [https://github.com/salesforce/ALBEF](https://github.com/salesforce/ALBEF).
- ViLT: We used the weights [ViLT-B/32 200k finetuned on NLVR2](https://github.com/dandelin/ViLT/releases/download/200k/vilt_nlvr2.ckpt). 
- BLIP: We follow the commands in [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)  
- TCL: We follow the commands in [https://github.com/uta-smile/TCL](https://github.com/uta-smile/TCL).
- METER: We used the weights [METER-CLIP16-RoBERTa fine-tuned on NLVR2 (resolution: 288^2)](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_nlvr2.ckpt).

#### Visual Entailment

- ALBEF: We follow the commands in [https://github.com/salesforce/ALBEF](https://github.com/salesforce/ALBEF).
- TCL: We follow the commands in [https://github.com/uta-smile/TCL](https://github.com/uta-smile/TCL).
- METER: We used the weights [METER-CLIP16-RoBERTa fine-tuned on SNLI-VE (resolution: 384^2)](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_snli.ckpt).

#### Image Captioning

- BLIP: We used the weights [BLIP w/ ViT-L](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth) .
- GRIT: We used the weights in [https://github.com/JialianW/GRiT](https://github.com/JialianW/GRiT).

#### Text-to-image Generation

- Stable Diffusion: We used the weights in [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
- GLIDE: We used the weights in [https://github.com/bumptech/glide](https://github.com/bumptech/glide).

