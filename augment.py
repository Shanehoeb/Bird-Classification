
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import sys

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog




# Configuration
cfg = get_cfg()
cfg.merge_from_file("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # threshold 
cfg.MODEL.ROI_HEADS.NMS = 0.4  
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
model = DefaultPredictor(cfg)


def fb_rcnn(img):
  # Modify colour channels
  img = np.array(img) 
  img = img[:, :, ::-1].copy() 
  with torch.no_grad():
          detections = model(img)["instances"]
  if len(detections.scores)>0:
    index_birds = np.where(detections.pred_classes.cpu().numpy()==14)[0] #Bird class
    if len(index_birds)==0:
      img = ImageOps.mirror(Image.fromarray(img))
    else:
      bird = int(torch.max(detections.scores[index_birds],0)[1].cpu().numpy())
      [x1,y1,x2,y2]=detections.pred_boxes[index_birds][bird].tensor[0].cpu().numpy()  
      # Slight enlarge to capture bird
      x1, y1 = np.maximum(0,int(x1)-20), np.maximum(0,int(y1)-20)
      x2, y2 = np.minimum(x2+40,img.shape[1]), np.minimum(y2+40,img.shape[0])
      img = img[int(np.ceil(y1)):int(y2), int(np.ceil(x1)):int(x2), :]
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(img)
  else:
    img = ImageOps.mirror(Image.fromarray(img))
  return img




def detect_birds(model, input_folder, output_folder):
  for data_folder in list(os.listdir(input_folder)): # Crop all 
    non_cropped = 0
    non_cropped_names = []
    num_imgs = 0
    directory = input_folder+'/'+data_folder
  
    for folder in list(os.listdir(directory)): 
      size = len(list(os.listdir(directory+'/'+folder)))
      num_imgs += size
      os.makedirs(output_folder, exist_ok = True)
      os.makedirs(output_folder+'/'+data_folder+'/'+folder, exist_ok = True)

      img_paths = []          
      img_detections = [] 
                
      for img_path in list(os.listdir(directory+'/'+folder)):
        img = cv2.imread(directory+'/'+folder+'/'+img_path)
        with torch.no_grad():
          detections = model(img)["instances"]
        img_paths.append(directory+'/'+folder+'/'+img_path)
        img_detections.append(detections)

      # Save cropped images
      for (path, detections) in (zip(img_paths, img_detections)):
        img = np.array(Image.open(path))
        if len(detections.scores)>0:
          index_birds = np.where(detections.pred_classes.cpu().numpy()==14)[0] # 14 is the default class number for bird
          if len(index_birds)==0:
            non_cropped_names.append(path)
            non_cropped += 1
            path = path.split("/")[-1]
            plt.imsave(output_folder+'/'+data_folder+'/'+folder+'/'+path, np.array(ImageOps.mirror(Image.fromarray(img))), dpi=1000)
            plt.close()  
            continue
          bird = int(torch.max(detections.scores[index_birds],0)[1].cpu().numpy())
          [x1,y1,x2,y2]=detections.pred_boxes[index_birds][bird].tensor[0].cpu().numpy()
          count=1
          x1, y1 = np.maximum(0,int(x1)-20), np.maximum(0,int(y1)-20)
          x2, y2 = np.minimum(x2+40,img.shape[1]), np.minimum(y2+40,img.shape[0])
          img = img[int(np.ceil(y1)):int(y2), int(np.ceil(x1)):int(x2), :]

          # Save Images
          path = path.split("/")[-1]
          plt.imsave(output_folder+'/'+data_folder+'/'+folder+'/'+path, img, dpi=1000)
          plt.close()   
          




