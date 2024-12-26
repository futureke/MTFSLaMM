from models.VLE import VLEModel, VLEProcessor
from PIL import Image
import torch
from transformers import AutoImageProcessor
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from mi_estimators import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
import argparse
from PIL import Image
from architecture import encoder, decoder, mi
import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import json
import os
from architecture import encoder, decoder, mi
from utils import vars_from_scopes, gaussian_nll
model_name="VLE-main/hfl/vle-base"

json_path = "VLE-main/captions/captions_val2017.json"
img_path = "VLE-main/pics/val2017/"

with open(json_path, "r") as f:
    json_file = json.load(f)
#print(json_file['annotations'])
images=[]
text=[]
for filename in os.listdir(img_path):
    file_path = os.path.join(img_path, filename)
    images.append(Image.open(file_path)) 

# 数据加载器 

for i in json_file['annotations']:  
    text.append(i['caption'])
print(images[0])
print(text[0])
#images = [Image.open('VLE-main/pics/val2017')] 
#text = ["There are dogs on the grass."]
model = VLEModel.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)
vle_processor = VLEProcessor.from_pretrained(model_name)
inputs = image_processor(images=images[0], return_tensors="pt")
image_features=model.get_image_features(**inputs)
print(image_features.shape)
print("image_features:", image_features)
multimodal_inputs = vle_processor(text=text[0],images=images[0], return_tensors='pt',padding=True)
print("vle_processor:",vle_processor)
print(multimodal_inputs['pixel_values'])
print(multimodal_inputs)
#forward
vle_output = model(**multimodal_inputs)
print(vle_output[2])
print(vle_output[2].shape)

input1=image_features.reshape(197,768)

input2=vle_output[2].reshape(197,768)
print(input1)
print(input1.type)
print(input2)
print(input2.type) 
input11=input1.detach().numpy()
input22=input2.detach().numpy()
# input2=torch.cuda().FloatTensor(input2)
# print(input1)
# print(input2)
#互信息计算
class GaussianSampler(nn.Module):
    def __init__(self, dim, para_list = None):
        super(GaussianSampler, self).__init__()
        self.dim = dim
        if para_list is None:
            para_list = [0.55] * dim
        self.p_theta_ = torch.nn.Parameter(torch.tensor(para_list, requires_grad = True))
        
    def get_trans_mat(self):
        p_theta = self.p_theta_.cuda().unsqueeze(-1)
        #p_theta = torch.softmax(p_theta, dim = 0)

        trans_row1 = torch.cat((torch.sin(p_theta),torch.cos(p_theta)), dim=-1).unsqueeze(-1)
        trans_row2 = torch.cat((torch.cos(p_theta),torch.sin(p_theta)), dim=-1).unsqueeze(-1)  #[dim, 2,1]
        return torch.cat((trans_row1, trans_row2), dim=-1)  #[dim,2,2]

    def gen_samples(self, num_sample, cuda = True):
        noise= torch.randn(self.dim,num_sample,2).cuda()
        trans_mat = self.get_trans_mat()
        samples = torch.bmm(noise, trans_mat).transpose(0,1) #[dim, nsample, 2]
        if not cuda:
            samples = samples.cpu().detach().numpy()
            samples[:,:,0]=input11
            samples[:,:,1]=input22
        # else: 
        #     samples[:,:,0]=input1
        #     samples[:,:,1]=input2
        # print(samples.shape)
        # print(samples.dtype)
        # print(samples[:,:,0].shape)
        
        return samples[:,:,0], samples[:,:,1] 

    def get_covariance(self):
        p_theta = self.p_theta_.cuda()
        return (2.*torch.sin(p_theta)*torch.cos(p_theta))

    def get_MI(self):
        rho = self.get_covariance()
        return -1./2.*torch.log(1-rho**2).sum().item()
        #return -self.dim /2.*torch.log(1-rho**2 / 2).sum().item()
lr = 1e-4
batch_size = 100 
num_iter = 15000
sample_dim = 768
hidden_size = 5
estimator_name = "CLUB"  
sampler = GaussianSampler(sample_dim)
#print("The corvariance of Gaussian is {}".format(sampler.get_covariance().cpu().detach().numpy()))
x_sample, y_sample = sampler.gen_samples(197, cuda = False)

mi_estimator = eval(estimator_name)(sample_dim, sample_dim, hidden_size).cuda()

sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr = lr)
mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr = lr)

mi_true_values = []
mi_est_values = []
min_mi = 100.

for i in range(num_iter):
    sampler.train()
    mi_estimator.eval()
    x_samples, y_samples = sampler.gen_samples(batch_size)
    if i==0:
        print(x_samples.shape)
    sampler_loss = mi_estimator(x_samples, y_samples)
    sampler_optimizer.zero_grad()
    sampler_loss.backward() # retain_graph=True)
    sampler_optimizer.step()

    for j in range(5):
        mi_estimator.train()
        x_samples, y_samples = sampler.gen_samples(batch_size)
        mi_loss = mi_estimator.learning_loss(x_samples, y_samples)
        mi_optimizer.zero_grad()
        mi_loss.backward()
        mi_optimizer.step()

    mi_true_values.append(sampler.get_MI())
    mi_est_values.append(mi_estimator(x_samples, y_samples).item())
    if i % 100 ==0:
        print("step {}, true MI value {}".format(i, sampler.get_MI()))