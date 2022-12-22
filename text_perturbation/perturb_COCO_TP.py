#!/usr/bin/env python
# coding: utf-8

import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from eda import *
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import random
random.seed(1234)


back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en'
)

rate_chunk = [1]    

def perturb_back_trans_json(annotation):
    
    for j in range(0, len(annotation)):       
        print(j)       
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]        
            aug_sentences = back_translation_aug.augment(tmp)
            if aug_sentences==None:
                aug_sentences = tmp
                    
            annotation[j]['caption'][i] = aug_sentences
    print("Perturbation finished for one file")
                
    return annotation   

for rate in rate_chunk:
    
    method = str('back_trans')   
    directory = "./annotation_%s/%s/"%(method,rate)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = perturb_back_trans_json(annotation)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))



rate_chunk = [1,2,3,4,5,6,7]


## Keyboard Augmenter 
def perturb_KeyboardAug_json(annotation, ratio):
    
    char_ratio=0.05*ratio
    aug = nac.KeyboardAug(aug_word_p=char_ratio)
    
    for j in range(0, len(annotation)):       
        print(j)       
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]        
            aug_sentences = aug.augment(tmp)
            if aug_sentences==None:
                aug_sentences = tmp
                    
            annotation[j]['caption'][i] = aug_sentences
    print("Perturbation finished for one file")
                
    return annotation    

## OCR Augmenter 
def perturb_OcrAug_json(annotation, ratio):
    
    char_ratio=0.05*ratio
    aug = nac.OcrAug(aug_word_p=char_ratio)
    
    for j in range(0, len(annotation)):       
        print(j)       
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]        
            aug_sentences = aug.augment(tmp)
            if aug_sentences==None:
                aug_sentences = tmp
                    
            annotation[j]['caption'][i] = aug_sentences
    print("Perturbation finished for one file")
                
    return annotation        

## Random Augmenter
def perturb_RandomCharAug_json(input_file, action, ratio):
    
    char_ratio=0.05*ratio   
    aug = nac.RandomCharAug(action, aug_word_p=char_ratio)
    
    for j in range(0, len(annotation)):       
        print(j)       
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]        
            aug_sentences = aug.augment(tmp)
            if aug_sentences==None:
                aug_sentences = tmp
                    
            annotation[j]['caption'][i] = aug_sentences
    print("Perturbation finished for one file")
                
    return annotation        



char_rate_chunk = [1,2,3,4,5,6,7]

for rate in char_rate_chunk:
    
    method = str('KeyboardAug')   
    directory = "./annotation_%s/%s/"%(method,rate)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = perturb_KeyboardAug_json(annotation, rate)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))


for rate in char_rate_chunk:
    
    method = str('OcrAug')   
    directory = "./annotation_%s/%s/"%(method,rate)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = perturb_OcrAug_json(annotation, rate)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))


action_chunk = ['insert','substitute','swap','delete']    

for action in action_chunk:
    
    for rate in char_rate_chunk:
        
        method = str('RandomCharAug_')+ action   
        directory = "./annotation_%s/%s/"%(method,rate)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
        os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
        annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
        print(len(annotation))
        new_annotation = perturb_RandomCharAug_json(annotation, action, rate)
        
        with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
            json.dump(new_annotation, f)
        print("finish %s"%(method))

'''

'''
from styleformer import Styleformer
import torch
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1234)

def text_style_perturb(annotation, style, style_value):
    
    sf = Styleformer(style = style_value)
    
    writer2 = open('coco_scores/coco_test_score_%s.txt'%(method), 'w')
    writer3 = open('coco_times/coco_test_times_%s.txt'%(method), 'w')
    
    for j in range(0, len(annotation)):
        
        print(j)
        
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]
            
            times= 0 
            while times <100:
                aug_sentences = sf.transfer(tmp)
                if aug_sentences==None:
                    aug_sentences = tmp
                #print("aug_sentences", aug_sentences)
                embeddings_aug = model.encode(aug_sentences)
                embeddings_base = model.encode(tmp)
                similarity_score = float(util.cos_sim(embeddings_aug, embeddings_base))
                if similarity_score < 0.9:
                    times = times+1
                    #print("perturb again")
                else:
                    #print(similarity_score)
                    break
                    
            annotation[j]['caption'][i] = aug_sentences
            
            writer2.write(str(similarity_score) + '\n')
            writer3.write(str(times) + '\n') 
            
    writer2.close()
    writer3.close()
    print("Perturbation finished for one file")
                
    return annotation

style_rate_chunk = [1]


for rate in style_rate_chunk:
    
    method = str('formal')   
    style_value = 0
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = text_style_perturb(annotation, method, style_value)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))


for rate in style_rate_chunk:
    
    method = str('casual')   
    style_value = 1
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = text_style_perturb(annotation, method, style_value)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))
    
for rate in style_rate_chunk:
    
    method = str('passive')   
    style_value = 2
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = text_style_perturb(annotation, method, style_value)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))
    
for rate in style_rate_chunk:
    
    method = str('active')   
    style_value = 3
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = text_style_perturb(annotation, method, style_value)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s"%(method))


def eda_perturb(method,annotation,alpha_sr,alpha_ri,alpha_rs,p_rd,num_aug):
    
    writer2 = open('coco_scores/coco_test_score_%s_%s.txt'%(method, alpha_sr), 'w')
    writer3 = open('coco_times/coco_test_times_%s_%s.txt'%(method, alpha_sr), 'w')
    
    for j in range(0, len(annotation)):
        
        print(j)
        
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]
            
            times= 0 
            while times <100:
                aug_sentences = eda(tmp, alpha_sr,alpha_ri,alpha_rs,p_rd,num_aug)
                
                embeddings_aug = model.encode(aug_sentences)
                embeddings_base = model.encode(tmp)
                similarity_score = float(util.cos_sim(embeddings_aug, embeddings_base))
                if similarity_score < 0.9:
                    times = times+1
                    #print("perturb again")
                else:
                    #print(similarity_score)
                    break
                    
            annotation[j]['caption'][i] = aug_sentences[0]
            
            writer2.write(str(similarity_score) + '\n')
            writer3.write(str(times) + '\n') 
            
    writer2.close()
    writer3.close()
    print("Perturbation finished for one file")
                
    return annotation

    
import random

random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']

def insert_punctuation_marks(sentence, punc_ratio):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line


def insert_punc(input_file,ratio):
    
    writer2 = open('coco_scores/coco_test_score_ip_' + str(ratio) + '.txt', 'w')
    writer3 = open('coco_times/coco_test_times_ip_' + str(ratio) + '.txt', 'w')
    
    
    for j in range(0, len(annotation)):  
        
        print(j)
        
        for i in range(0,5):
            tmp = annotation[j]['caption'][i]
            
            times= 0 
            while times <100:
                aug_sentences = insert_punctuation_marks(tmp, ratio)
                
                embeddings_aug = model.encode(aug_sentences)
                embeddings_base = model.encode(tmp)
                similarity_score = float(util.cos_sim(embeddings_aug, embeddings_base))
                if similarity_score < 0.9:
                    times = times+1
                    #print("perturb again")
                else:
                    #print(similarity_score)
                    break
            
            annotation[j]['caption'][i] = aug_sentences
            
            writer2.write(str(similarity_score) + '\n')
            writer3.write(str(times) + '\n') 
            
    writer2.close()
    writer3.close()
    print("Perturbation finished for one file")
                
    return annotation

    
rate_chunk = [1,2,3,4,5]
rate_chunk = [6,7]

#ip
for rate in rate_chunk:
    
    method = str('ip')    
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    current_rate = 0.05*rate
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = insert_punc(annotation,ratio=current_rate)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s %s"%(method,current_rate))
    
#sr    
for rate in rate_chunk:
    
    method = str('sr')    
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    current_rate = 0.05*rate
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = eda_perturb(method, annotation,alpha_sr=current_rate, alpha_ri=0.0, alpha_rs=0.0, p_rd=0.0, num_aug=1)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s %s"%(method,current_rate))



#ri
for rate in rate_chunk:
    
    method = str('ri')    
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    current_rate = 0.05*rate
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = eda_perturb(method, annotation,alpha_sr=0.0, alpha_ri=current_rate, alpha_rs=0.0, p_rd=0.0, num_aug=1)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s %s"%(method,current_rate))


#rs
for rate in rate_chunk:
    
    method = str('rs')    
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    current_rate = 0.05*rate
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = eda_perturb(method, annotation,alpha_sr=0.0, alpha_ri=0.0, alpha_rs=current_rate, p_rd=0.0, num_aug=1)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s %s"%(method,current_rate))

    
#rd
for rate in rate_chunk:
    
    method = str('rd')    
    #os.mkdir("./annotation_%s/%s/"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_train.json ./annotation_%s/%s/coco_karpathy_train.json"%(method,rate))
    os.system("cp -r ../original_annotation/coco_karpathy_val.json ./annotation_%s/%s/coco_karpathy_val.json"%(method,rate))
        
    current_rate = 0.05*rate
    annotation = json.load(open('../original_annotation/coco_karpathy_test.json','r'))
    print(len(annotation))
    new_annotation = eda_perturb(method, annotation,alpha_sr=0.0, alpha_ri=0.0, alpha_rs=0.0, p_rd=current_rate, num_aug=1)
        
    with open('./annotation_%s/%s/coco_karpathy_test.json'%(method,rate), 'w') as f:
        json.dump(new_annotation, f)
            
    print("finish %s %s"%(method,current_rate))



