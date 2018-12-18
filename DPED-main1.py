#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
import scipy.misc
from easydict import EasyDict as edict
from DPED import *
from utils import *
from ops import *

config = edict()
# training parameters
config.batch_size = 50
config.patch_size = 100
config.mode = "RGB" #YCbCr
config.channels = 3
config.content_layer = 'relu5_4'
config.learning_rate = 1e-4
config.augmentation = True #data augmentation (flip, rotation)

# weights for loss
config.w_color = 1.2 # gaussian blur + mse (originally 0.1)
config.w_texture = 1 # gan (originally 0.4)
config.w_content = 2 # vgg19 (originally 1)
config.w_tv = 1/400 # total variation (originally 400)

# directories
config.dataset_name = "iphone"
config.train_path_phone = os.path.join("/home/sun/Disk/Photo-enhancer/dped",str(config.dataset_name),"training_data",str(config.dataset_name),"*.jpg")
config.train_path_dslr = os.path.join("/home/sun/Disk/Photo-enhancer/dped",str(config.dataset_name),"training_data/canon/*.jpg")
config.test_path_phone_patch = os.path.join("/home/sun/Disk/Photo-enhancer/dped",str(config.dataset_name),"test_data/patches",str(config.dataset_name),"*.jpg")
config.test_path_dslr_patch = os.path.join("/home/sun/Disk/Photo-enhancer/dped",str(config.dataset_name),"test_data/patches/canon/*.jpg")
config.test_path_phone_image = os.path.join("/home/sun/Disk/Photo-enhancer/sample_images/original_images",str(config.dataset_name),"*.jpg")
config.test_path_dslr_image = os.path.join("/home/sun/Disk/Photo-enhancer/sample_images/original_images/canon/*.jpg")
config.sample_dir = "samples"
config.checkpoint_dir = "checkpoint"
config.vgg_dir = "/home/sun/Disk/Photo-enhancer/vgg_pretrained/imagenet-vgg-verydeep-19.mat"
config.log_dir = "logs"

if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)
if not os.path.exists(config.sample_dir):
    os.makedirs(config.sample_dir)
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)


# In[2]:


# load dataset
dataset_phone, dataset_dslr = load_dataset(config)


# In[2]:


# build DPED model
tf.reset_default_graph()
# uncomment this when only trying to test the model
# dataset_phone = []
# dataset_dslr = []
sess = tf.Session()
model = DPED(sess, config, dataset_phone, dataset_dslr)


# In[10]:


# pretrain discriminator with (phone, dslr) pairs
model.pretrain_discriminator(load = False)


# In[3]:


# test discriminator performance for (phone, dslr) pair
model.test_discriminator(200, load = True)


# In[12]:


# train generator & discriminator together
model.train(load = True)


# In[3]:


# test trained model
model.test_generator(200, 14, load = True)


# In[13]:


# save trained model
model.save()


# In[ ]:




