#Tạo file train.txt, valid.txt theo đoạn code

import os
import numpy as np
#"obj" là tên thư mục chứa cả ảnh và file annotation.
lst_files = os.listdir("data/obj/")
lst_images = []

for file in lst_files:
  if ".txt" not in file:
    lst_images.append(file)
    
#Tách 200 ảnh ra làm tập validation  
random_idx = np.random.randint(0, len(lst_images), 200)

#Tạo file train.txt được đặt trong thư mục darknet/data
with open("data/train.txt","w") as f:
  for idx in range(len(lst_images)):
    if idx not in random_idx:
      f.write("data/obj/"+lst_images[idx]+"\n")
      
#Tạo file valid.txt được đặt trong thư mục darknet/data
with open("data/valid.txt","w") as f:   
    for idx in random_idx: