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
model = VLEModel.from_pretrained(model_name)
print(model)