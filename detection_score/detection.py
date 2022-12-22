import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv
import pandas as pd

def load(image_name):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(image_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    


if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='')
    parser.add_argument('--input_path', default='')
    parser.add_argument('--output_path', default='')
    args = parser.parse_args()

    method = args.method
    input_path = args.input_path
    output_path = args.output_path
    print("method:", method)
    
    config_file = "configs/pretrain/glip_Swin_L.yaml"
    weight_file = "MODEL/glip_large_model.pth"


    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    
    
    
    for num_image in range(0,16):
        print("current image:", num_image)

        num_all = 0
        score_all = 0.0

        with open('cat_per_img.txt', 'r') as train_orig:
        
            for ori_index, prompt in enumerate(train_orig):

                text_index = ori_index+1
            
                caption = prompt
            
                image_name = '%s/text_%s/%s.png'%(input_path,text_index,num_image)
                image = load(image_name)
                result, top_predictions, new_labels = glip_demo.run_on_web_image(image, caption, 0.7)
            
                    
                # num
                num = len(top_predictions.bbox)
                num_all = num_all + num
                      
                # scores
                scores= top_predictions.get_field("scores").tolist()
                narry_scores = np.zeros(100)
                for i in range(0, len(scores)):
                    score_all = score_all + float(scores[i])
                    narry_scores[i] = format(float(scores[i]), '.4f')
                        
        print("num_all:", num_all)     
        print("score_all:", score_all)  
        average_score = score_all/num_all 
        print("average score:", average_score) 
        
        file_object = open('%s/scores_0.7_%s.txt'%(output_path,method), 'a+', encoding="utf-8")
        file_object.write(str(num_all) + '\n')
        file_object.write(str(score_all) + '\n')
        file_object.write(str(average_score) + '\n')
        file_object.close()