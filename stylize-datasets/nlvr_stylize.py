#!/usr/bin/env python
import os
import argparse
from function import adaptive_instance_normalization
import net
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


severity_chunk = [1,2,3,4,5]

for tmp in severity_chunk:
    current_severity = tmp
    current_alpha = current_severity*0.2
    print("current_severity:",current_severity)
    os.system("python stylize.py --content-dir='./nlvr_dev/' --output-dir='./nlvr/dev-IP/nlvr_IP_stylize_%s/' --alpha=%s"%(current_severity,current_alpha))
    print("finish stylize %s"%tmp)


for tmp in severity_chunk:
    current_severity = tmp
    current_alpha = current_severity*0.2
    print("current_severity:",current_severity)
    os.system("python stylize.py --content-dir='./nlvr_test1/' --output-dir='./nlvr/test1-IP/nlvr_IP_stylize_%s/' --alpha=%s"%(current_severity,current_alpha))
    print("finish stylize %s"%tmp)