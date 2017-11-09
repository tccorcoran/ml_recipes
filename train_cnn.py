import random
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.utils.training_utils import multi_gpu_model
import argparse

import cv2
import numpy as np
import tensorflow as tf
from pdb import set_trace

def random_adjust_brightness(img, scale=(-30,30)):
    """Add a random constant to accross all channels of the image
    args
        scale: interval where the random constant lives
    returns:
        Augmented image as float32 dtype
    """
    scale = random.randint(scale[0],scale[1])
    scaled_img = img.astype(np.float32) + scale
    scaled_img[scaled_img > 255] = 255
    scaled_img[scaled_img < 0] = 0
    return scaled_img

def count_number_files(dir):
    """Counts number of files in a directory
    Note: counts all files, not just images
    """
    total = 0
    for root, dirs, files in os.walk(dir):
        total += len(files)
    return total

def count_number_folders(dir):
    """ Counts number of dirs in a directory
    """
    return sum(os.path.isdir(os.path.join(dir,i)) for i in os.listdir(dir))

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, nargs='?',default=[300,300])
parser.add_argument('--train_dir',type=str,required=True)
parser.add_argument('--test_dir',type=str,required=True)
parser.add_argument('--model_dir',type=str,required=True)
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--epochs',type=int,default=1000)
args = parser.parse_args()
    
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
    
train_size = count_number_files(args.train_dir)
test_size = count_number_files(args.test_dir)

num_classes = count_number_folders(args.train_dir)
print num_classes
if num_classes != count_number_folders(args.test_dir):
    raise ValueError("Number of classes in test and train do not match")
img_h = args.img_size[0]
img_w = args.img_size[1]

#with tf.device("/cpu:0"):
model = Xception(include_top=True, weights=None, input_shape=(img_h,img_w,3), pooling=None, classes=num_classes)

#model = multi_gpu_model(model, gpus=2)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

print model.summary()


model_name = os.path.join(args.model_dir,"doc_detect.{epoch:02d}-{val_loss:.2f}.hdf5")
checkpointer = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

train_datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=20)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(img_w, img_w),  
        batch_size=args.batch_size) 


validation_generator = test_datagen.flow_from_directory(
                        args.test_dir,
                        target_size=(img_w, img_w),
                        batch_size=args.batch_size)

print train_generator.class_indices

model.fit_generator(
        train_generator,
        steps_per_epoch=train_size // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=test_size // args.batch_size,
        callbacks=[checkpointer])    
