# -*- coding: utf-8 -*-
from typing import Callable, Tuple,  Any
from functools import partial
import torch
import json
import os
import albumentations as albu
import cv2
import numpy as np
import re
from torch import nn
import cv2
from PIL import Image
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Callable, Dict, Mapping, Tuple, Optional, Union

C2LP = 'class2label.json'
encodername = 'efficientnet'
MW = 'finalw.pth'

# +
ENCODERS = {
    
    'efficientnet': {
        'features': 1408,
        'init_op': EfficientNet.from_name('efficientnet-b2', num_classes=131),
    },

}


# -

class SignsClassifier(nn.Module):
    """
    A model for classifying signs.
    """

    def __init__(self, encoder_name: str, n_classes: int, dropout_rate: float = 0.0):
        """Initializing the class.

        :param encoder_name: name of the network encoders
        :param n_classes: number of output classes
        :param dropout_rate: dropout rate
        """
        super().__init__()
        self.encoder = ENCODERS[encoder_name]['init_op']
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        
        self.fc = Linear(ENCODERS[encoder_name]['features'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Getting the model prediction.

        :param x: input batch tensor
        :return: prediction
        """
        x = self.encoder(inputs=x)
        
        return x


def load_json_file(path: str) -> Any: # Loading JSON File
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def prep(img, img_size: Tuple[int, int] = (224, 224)) -> Callable: # Preparing Image To Enter The Model
    valid_transform = [
        albu.Resize(img_size[0], img_size[1]),
    ]
    img = albu.Compose(valid_transform)(image=img)['image']
    img = img.astype(np.float32)
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    img -= np.array([0.485, 0.456, 0.406])[:, None, None]
    img /= np.array([0.229, 0.224, 0.225])[:, None, None]

    return img

def gen_pred(imgss, class2label):
    model = SignsClassifier('efficientnet', 131)
    sdict = torch.load(MW, map_location='cuda')
    model.load_state_dict(sdict['state_dict'])  # Loading 'best.pth'
    model.eval() # Setting The Module Into The Evaluation Mode
    label2class = {v : k for k, v in class2label.items()} # Converting Prediction Into Actual Class Labels
    img = Image.open(imgss) # Opening The File Using PIL
    img = np.array(img) # Converting The Image To A NumPy Array
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) # Converting Image From One Color Space To Another
    im = prep(img, (224, 224)) # Applying The Prep Function
    im = torch.from_numpy(im) # NumPy Array -> Tensor
    im = im.unsqueeze(0).cuda()
    pred = model(im) # Applying The Earlier Model Variable
    pred = nn.LogSoftmax(dim=1)(pred) # Using SoftMax Activation
    pred = pred.argmax(dim=-1).cpu().numpy()[0]
    pred = label2class[pred] # Converting Prediction To Class ID
    #en_pred = en_pred[pred] # Returning Prediction In English 
    #ru_pred = ru_pred[pred] # Returning Prediction In Russian 

    return pred#, ru_pred, en_pred


#translate = load_json_file('./app/rus_direction_to_eng_direction.json') 


def to_eng(pred, translate):
    newpred = []
    for i in range(len(pred.split(','))):
        if i != 0:
            newpred.append(',')
        for j in range(len(pred.split(',')[i].split('+'))):
            if j != 0:
                newpred.append('+')
            newpred.append(translate[pred.split(',')[i].split('+')[j]])
            
        
    string = ''
    for i in newpred:
        string += i
    return string


def printit(pred, translate, isen):
    if isen:
        res = ['1 line: ']
        splt = pred.split(',')
        for i in range(len(splt)):
            if i != 0:
                res.append(f'\n{i+1} line: ')
        for j in range(len(splt[i].split('+'))):
            if j != 0:
                res.append(' and ')
            res.append(to_eng(splt[i].split('+')[j], translate))
        ans = ''
        for i in res:
            ans += i
        return ans
    else:
        await message.answer('Успешно загружено.')
        res = ['1 полоса: ']
        splt = pred.split(',')
        for i in range(len(splt)):
          if i != 0:
            res.append(f'\n{i+1} полоса: ')
          for j in range(len(splt[i].split('+'))):
            if j != 0:
              res.append(' и ')
            res.append(splt[i].split('+')[j])
        ans = ''
        for i in res:
            ans += i
        return ans