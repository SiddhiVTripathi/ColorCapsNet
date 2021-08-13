import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow import keras

from sklearn.utils import shuffle
import pandas as pd


ds = tfds.load('div2k', split='train', shuffle_files=True)
df = tfds.as_dataframe(ds.take(800))

ds_validation= tfds.load('div2k', split= 'validation')
df_val = tfds.as_dataframe(ds_validation.take(100))

print("Loading data in df...")
df_val.drop('hr',inplace=True, axis=1)
df.drop('hr',inplace=True, axis=1)

def color2bw(x):
  return np.stack((cv.cvtColor(x, cv.COLOR_BGR2GRAY),)*3, axis=-1)

def color2lab(x):
  return cv.cvtColor(x, cv.COLOR_BGR2LAB)

def patching(gray_image, color_image,x_train_gray, x_train_color):
  count_img = 0
  count_total_patch = 0
  count_patch = 0
  count_img+=1
  for y in range(0, gray_image.shape[0], STRIDE):
      for x in range(0, gray_image.shape[1], STRIDE):
          gray_patch = gray_image[y:y+PATCH_SIZE,x:x+PATCH_SIZE,:]
          color_patch = color_image[y:y+PATCH_SIZE,x:x+PATCH_SIZE,:]
          if gray_patch.shape[0:2] != (PATCH_SIZE,PATCH_SIZE) or \
  color_patch.shape[0:2] != (PATCH_SIZE,PATCH_SIZE):
              continue
          count_patch += 1
          count_total_patch += 1
          if count_total_patch % SAMPLING != 0:
              continue
          x_train_gray.append(gray_patch)
          x_train_color.append(color_patch)
          print ('Processed ' + str(count_patch) + ' / ' + str(count_img))
print("Augmenting...")
df['lr_gray']=df['lr'].apply(color2bw)
df_val['lr_gray']=df_val['lr'].apply(color2bw)
df['lr_lab']=df['lr'].apply(color2lab)
df_val['lr_lab']=df_val['lr'].apply(color2lab)

PATCH_SIZE = 9 #8 #9 #128
STRIDE = 9 #8 #9 #128
SAMPLING = 4

x_train_color,x_train_gray=[],[]
x_val_color,x_val_gray = [],[]

print("Patching...")
for ind in df.index:
     patching(df['lr_gray'][ind], df['lr_lab'][ind],x_train_gray=x_train_gray,x_train_color=x_train_color)

OUT_PATH = '{0}/../train_{1}_{2}_{3}.npz'.format("",PATCH_SIZE,STRIDE,SAMPLING)
print("Patching...")
for ind in df.index:
     patching(df_val['lr_gray'][ind], df_val['lr_lab'][ind], x_train_color=x_val_color,x_train_gray=x_val_gray)

OUT_PATH_val = '{0}/../val_{1}_{2}_{3}.npz'.format("",PATCH_SIZE,STRIDE,SAMPLING)

x_train_gray = np.array(x_train_gray)
x_train_color = np.array(x_train_color)

# shuffling
x_train_gray,x_train_color = shuffle(x_train_gray,x_train_color)

print('Saving..')
np.savez(OUT_PATH, x_train_gray, x_train_color)
print('Saved..')

x_val_gray = np.array(x_val_gray)
x_val_color = np.array(x_val_color)

# shuffling
x_val_gray,x_val_color = shuffle(x_val_gray,x_val_color)

print('Saving..')
np.savez(OUT_PATH_val, x_val_gray, x_val_color)
print('Saved..')