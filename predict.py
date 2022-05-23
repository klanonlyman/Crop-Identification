#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from model import swin_large_patch4_window12_384_in22k as create_model
import numpy as np
import csv
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
def test():
    test_path = "./test"
    json_path = './class_indices.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 14
    img_size = 384
    data_transform = transforms.Compose([transforms.Resize([img_size,img_size]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    model = create_model(num_classes=num_classes).to(device)
    
    list_= ['0','1','2','3']
    for item in list_:
        print("model: %s\n"%item)
        model_name = 'model-%s'%item
        
        # load model weights
        model_weight_path = "./weights/%s.pth"%model_name
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        model.cuda()
        
        ensemble = pd.DataFrame(columns=['img','0','1','2','3','4','5','6'
                                         ,'7','8','9','10','11','12','13'])

        number=0
        file = os.listdir(test_path)
        for img_name in file:
            img_path = test_path + "/" + img_name
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            ensemble_output = output.numpy()
            pro = {'img':img_name,'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0
                                         ,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0}
            for y in range (14):
                pro[str(y)] = ensemble_output[y]
            
            ensemble = ensemble.append(pro,ignore_index=True)
            number=number+1
            if number%100==1:
                print("img:",number)
        ensemble.to_csv("./result/%s.csv"%str(item),index=False)
    file=os.listdir("result")
    img_label={}
    for name in file:
        file_name=os.path.join("result",name)
        df=pd.read_csv(file_name)
        for index in range(0,len(df)):
            sub=df.loc[index]
            img=sub["img"]
            if img not in img_label:
                img_label[img]=np.zeros((14)).astype("float32")
            for i in range(0,14):
                p=sub[str(i)]
                img_label[img][i]+=p
    with open("result.csv", "a+",newline="") as csvfile:
        writeCsv = csv.writer(csvfile)
        writeCsv.writerow(["image_filename","label"])
        for img in img_label:
            label=class_indict[str(np.argmax(softmax(img_label[img])))]
            writeCsv.writerow([img,label])
if __name__ == '__main__':
    test()
    
