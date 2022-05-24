#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
from PIL import Image
 
path="data"
floder=os.listdir(path)
for sub_floder in floder:
    floder_path=os.path.join(path,sub_floder)
    if not(os.path.exists(floder_path.replace("data","train"))):
        os.makedirs(floder_path.replace("data","train"))
    images=os.listdir(floder_path)
    for name in images:
        image_path=os.path.join(floder_path,name)
        img = Image.open(image_path)
        out = img.resize((384, 384))
        out.save(image_path.replace("data","train"))




