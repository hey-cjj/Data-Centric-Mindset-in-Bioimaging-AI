import cv2 
import tifffile as tf
import os
import numpy as np

src_dir = './data/origin'
save_dir = './data/train'

size = 224
overlap = 0

mean =  0
std = 0

for i,name in enumerate(os.listdir(src_dir)):
    path = os.path.join(src_dir,name)
    image = tf.imread(path)      

    z,h ,w = image.shape
    for d in range(0,z):
        for y in range(0,h-size,size):
            for x in range(0,w-size,size):
                image_crop = image[0,d,y:y+size,x:x+size]
                if image_crop.max() == 0:
                    continue
                if image_crop.shape != (224,224):
                    print(image_crop.shape)
                    break
                save_path = os.path.join(save_dir,name.split('.')[0]+f'_{d}_{y}_{x}.tiff')
                tf.imsave(save_path,image_crop)
        # save_path = os.path.join(save_dir,name.split('.')[0]+f'_{d}.tiff')
        # tf.imsave(save_path,image[0,d,:,:])

    image = image / 65535. # int16

    old_mean = mean
    mean1 = image[:,:,:].mean()
    mean1 = old_mean + 1/(i+1)*(mean - old_mean)

    old_std = std
    std1 = image[0,:,:,:].std()
    std1 = old_std + 1/(i+1)*(std - old_std)

   
print(mean)   #0.09402948325862738
print(std)    #0.04949164850403422
