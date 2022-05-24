#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_large_patch4_window12_384_in22k as create_model
from utils import read_split_data,one_epoch, evaluate

if __name__ == '__main__':
  data_path="train"
  weight_path="./swin_large_patch4_window12_384_22k.pth"
  device="cuda:0"
  num_classes=14
  epochs=30
  batch_size=2
  lr=2e-6
  
  
  device = torch.device(device if torch.cuda.is_available() else "cpu")
  if os.path.exists("./weights") is False:
      os.makedirs("./weights")
      
  tb_writer = SummaryWriter()
  train_images_path, train_images_label = read_split_data(data_path)
  img_size = 384
  data_transform = {
      "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
      }
  train_dataset = MyDataSet(images_path=train_images_path,
                                images_class=train_images_label,
                                transform=data_transform["train"])
  nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=train_dataset.collate_fn)
  
  model = create_model(num_classes=num_classes).to(device)
  
  
  if weight_path != "":
      assert os.path.exists(weight_path), "weights file: '{}' not exist.".format(args.weights)
      weights_dict = torch.load(weight_path, map_location=device)["model"]

      for k in list(weights_dict.keys()):
          if "head" in k:
              del weights_dict[k]
      print(model.load_state_dict(weights_dict, strict=False))
      
  pg = [p for p in model.parameters() if p.requires_grad]
  optimizer = optim.AdamW(pg, lr=lr, weight_decay=1e-7)
  for epoch in range(epochs):
      train_loss, train_acc = one_epoch(model=model,
                                              optimizer=optimizer,
                                              data_loader=train_loader,
                                              device=device,
                                              epoch=epoch)
  
  
  
      torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


# In[ ]:




