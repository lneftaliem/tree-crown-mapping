#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import h5py
import numpy as np
from PIL import Image
import rasterio
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import imageio
import os
import time
import rasterio.warp             # Reproject raster samples
from functools import reduce
from tensorflow.keras.models import load_model

from UNet import UNet
from losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from optimizers import adaDelta, adagrad, adam, nadam
from frame_info import FrameInfo
from dataset_generator import DataGenerator
from split_frames import split_dataset
from visualize import display_images

import json
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# Suppress warnings and logging
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Import tensorflow and check version
import tensorflow as tf
print(tf.__version__)


# In[2]:


# Required configurations (including the input and output paths) are stored in a separate file (such as config/UNetTraining.py)
# Please provide required info in the file before continuing with this notebook. 
 
import UNetTraining
# In case you are using a different folder name such as configLargeCluster, then you should import from the respective folder 
# Eg. from configLargeCluster import UNetTraining
config = UNetTraining.Configuration()


# In[3]:


#from PIL import Image

# Disable the decompression bomb protection
#Image.MAX_IMAGE_PIXELS = None

# Read all images/frames into memory
#frames = []

#my original
#all_files = os.listdir(config.base_dir)
#all_files_ndvi = [fn for fn in all_files if fn.startswith(config.ndvi_fn) and fn.endswith(config.image_type)]
#len(all_files_ndvi)
#for i, fn in enumerate(all_files_ndvi):
#    ndvi_img = rasterio.open(os.path.join(config.base_dir, fn))
#    pan_img = rasterio.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.pan_fn)))
#    read_ndvi_img = ndvi_img.read()
#    read_pan_img = pan_img.read()
#    comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
#    comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
#    annotation_im = Image.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.annotation_fn)))
#    annotation = np.array(annotation_im)
#    weight_im = Image.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.weight_fn)))
#    weight = np.array(weight_im)
#    f = FrameInfo(comb_img, annotation, weight)
#    frames.append(f)

#cursor:
from PIL import Image
import gc
import psutil
import numpy as np

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def read_and_downsample_image(src, scale_factor=0.25):
    """Read image and downsample it"""
    # Calculate new dimensions
    new_height = int(src.height * scale_factor)
    new_width = int(src.width * scale_factor)
    
    # Read and resample in one step using rasterio's built-in resampling
    data = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=rasterio.enums.Resampling.bilinear
    )
    return data

# Disable the decompression bomb protection
Image.MAX_IMAGE_PIXELS = None

# Read all images/frames into memory
frames = []

all_files = os.listdir(config.base_dir)
all_files_ndvi = [fn for fn in all_files if fn.startswith(config.ndvi_fn) and fn.endswith(config.image_type)]
print(f"Found {len(all_files_ndvi)} files to process")

for i, fn in enumerate(all_files_ndvi):
    print(f"\nProcessing image {i+1}/{len(all_files_ndvi)}: {fn}")
    print(f"Memory usage before processing: {get_memory_usage():.1f} MB")
    
    try:
        # Process NDVI image
        with rasterio.open(os.path.join(config.base_dir, fn)) as ndvi_img:
            print(f"Reading and downsampling NDVI image...")
            read_ndvi_img = read_and_downsample_image(ndvi_img)
            print(f"NDVI shape: {read_ndvi_img.shape}")
        
        # Process PAN image
        with rasterio.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.pan_fn))) as pan_img:
            print(f"Reading and downsampling PAN image...")
            read_pan_img = read_and_downsample_image(pan_img)
            print(f"PAN shape: {read_pan_img.shape}")
        
        # Combine images
        comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
        del read_ndvi_img, read_pan_img
        gc.collect()
        
        comb_img = np.transpose(comb_img, axes=(1,2,0))
        print(f"Combined shape: {comb_img.shape}")
        
        # Process annotation
        with Image.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.annotation_fn))) as annotation_im:
            # Resize annotation using PIL
            new_size = (comb_img.shape[1], comb_img.shape[0])
            annotation_im = annotation_im.resize(new_size, Image.NEAREST)
            annotation = np.array(annotation_im)
            print(f"Annotation shape: {annotation.shape}")
        
        # Process weight
        with Image.open(os.path.join(config.base_dir, fn.replace(config.ndvi_fn,config.weight_fn))) as weight_im:
            # Resize weight using PIL
            weight_im = weight_im.resize(new_size, Image.NEAREST)
            weight = np.array(weight_im)
            print(f"Weight shape: {weight.shape}")
        
        f = FrameInfo(comb_img, annotation, weight)
        frames.append(f)
        
        # Clean up
        del comb_img, annotation, weight
        gc.collect()
        
        print(f"Memory usage after processing: {get_memory_usage():.1f} MB")
        
    except Exception as e:
        print(f"Error processing file {fn}: {str(e)}")
        continue

print(f"\nFinal memory usage: {get_memory_usage():.1f} MB")
print(f"Successfully processed {len(frames)} frames")   


# In[4]:


# Delete existing JSON file if it exists to force a new split
if os.path.exists(config.frames_json):
    os.remove(config.frames_json)

# Now call split_dataset with all frames
training_frames, validation_frames, testing_frames = split_dataset(frames, config.frames_json, config.patch_dir)

annotation_channels = config.input_label_channel + config.input_weight_channel
# ... rest of existing generator code ...
training_frames, validation_frames, testing_frames  = split_dataset(frames, config.frames_json, config.patch_dir)

annotation_channels = config.input_label_channel + config.input_weight_channel
train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)
val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)
test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)


# Define your configuration variables if they aren't already set
#annotation_channels = config.input_label_channel  # Assumed to be [2] or whatever your annotation channel is

# Filter out frames that don't have any annotations
#annotated_frame_indices = filter_frames_with_annotations(frames, annotation_channels)

# Ensure some frames were filtered
#if not annotated_frame_indices:
#    raise ValueError("No annotated frames found. Check your data.")

# Now split the filtered annotated frames into training, validation, and testing sets
#training_frames, validation_frames, testing_frames = split_dataset(
#    annotated_frame_indices, config.frames_json, config.patch_dir
#)

# Proceed with training using only annotated frames
#train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames, annotation_channels, augmenter='iaa').random_generator(config.BATCH_SIZE, normalize=config.normalize)
#val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames, annotation_channels, augmenter=None).random_generator(config.BATCH_SIZE, normalize=config.normalize)
#test_generator = DataGenerator(config.input_image_channel, config.patch_size, testing_frames, frames, annotation_channels, augmenter=None).random_generator(config.BATCH_SIZE, normalize=config.normalize)


# In[5]:


for _ in range(1):
    train_images, real_label = next(train_generator)
    ann = real_label[:,:,:,0]
    wei = real_label[:,:,:,1]
    #overlay of annotation with boundary to check the accuracy
    #5 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    overlay = ann + wei
    overlay = overlay[:,:,:,np.newaxis]
    display_images(np.concatenate((train_images,real_label, overlay), axis = -1))


# In[6]:


#OPTIMIZER = adaDelta
#LOSS = tversky 
OPTIMIZER = tf.keras.optimizers.Adadelta()  # Make sure to instantiate it
LOSS = tversky 

#Only for the name of the model in the very end
OPTIMIZER_NAME = 'AdaDelta'
LOSS_NAME = 'weightmap_tversky'

# Declare the path to the final model
# If you want to retrain an exising model then change the cell where model is declared. 
# This path is for storing a model after training.

timestr = time.strftime("%Y%m%d-%H%M")
chf = config.input_image_channel + config.input_label_channel
chs = reduce(lambda a,b: a+str(b), chf, '')


if not os.path.exists(config.model_path):
    os.makedirs(config.model_path)
model_path = os.path.join(config.model_path,'trees_{}_{}_{}_{}_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0]))

# The weights without the model architecture can also be saved. Just saving the weights is more efficent.

# weight_path="./saved_weights/UNet/{}/".format(timestr)
# if not os.path.exists(weight_path):
#     os.makedirs(weight_path)
# weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
# print(weight_path)


# In[7]:


# CORRECTED SOLUTION FOR H5 WEIGHTS FILE
# SIMPLE SOLUTION FOR LAYER COUNT MISMATCH
# Sahel/Sudan U-Net weights of Brandt et al. (2020), obtained from the authors
pretrained_weights_path = os.environ.get('PRETRAINED_WEIGHTS', './saved_models/UNet/sahel_sudan_v501_norm40_weights.h5')

print("🔧 Loading H5 weights with by_name=True (allows architectural differences)...")

try:
    # Create model with your preferred layer count
    model = UNet([config.BATCH_SIZE, *config.input_shape], config.input_label_channel, layer_count=64)

    # Load weights with by_name=True - this is the key parameter!
    # It allows loading weights even if architectures don't match exactly
    model.load_weights(pretrained_weights_path, by_name=True)

    print("Successfully loaded H5 weights using by_name=True!")
    print("This means some layers matched by name and were loaded,")
    print("   while unmatched layers kept their random initialization.")

except Exception as e:
    print(f"Failed to load weights: {e}")
    print("This usually means the model architectures are too different.")
    print("Creating model without pretrained weights...")

    # Fallback: create model without weights
    model = UNet([config.BATCH_SIZE, *config.input_shape], config.input_label_channel)
    print("Created model without pretrained weights (training from scratch)")

model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

print("Model ready for training!")

# Show model summary to verify
print("\n Model Architecture Summary:")
model.summary()


#old
#model = UNet([config.BATCH_SIZE, *config.input_shape], config.input_label_channel, weight_file=pretrained_weights_path)
#model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])


# In[8]:


# Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Update model path
checkpoint = ModelCheckpoint(model_path.replace('.h5', '.keras'), monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only=False)

# Reduce learning rate when a metric has stopped improving
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                   patience=4, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=4, min_lr=1e-16)

# TensorBoard callback
log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}'.format(timestr, OPTIMIZER_NAME, LOSS_NAME, chs, config.input_shape[0]))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, 
                          write_images=False, embeddings_freq=0, update_freq='epoch')

callbacks_list = [checkpoint, tensorboard] # reduceLROnPlat is not required with adaDelta

# Add this new code block for class weights calculation
print("Calculating class weights...")

# Get a batch of data to analyze class distribution
sample_batches = 10
tree_pixels = 0
total_pixels = 0

for _ in range(sample_batches):
    images, labels = next(train_generator)
    tree_pixels += np.sum(labels[:,:,:,0])  # Sum the tree class pixels
    total_pixels += labels[:,:,:,0].size

background_pixels = total_pixels - tree_pixels
ratio = background_pixels / tree_pixels
print(f"Tree to background pixel ratio: 1:{ratio:.2f}")


# Calculate balanced class weights
class_weight = {
    0: 1.0,  # background class
    1: ratio  # tree class
}

print(f"Using class weights: {class_weight}")


# In[ ]:


# Import Keras backend
#from tensorflow.keras import backend as K
# Limit GPU memory growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)romma
        

# Training loop with session clearing
#for epoch in range(config.NB_EPOCHS):
#    loss_history = model.fit(train_generator, 
#                             steps_per_epoch=config.MAX_TRAIN_STEPS, 
#                             epochs=config.NB_EPOCHS,  # Train for 1 epoch at a time
#                             validation_data=val_generator,
#                             validation_steps=config.VALID_IMG_COUNT,
#                             callbacks=callbacks_list
                             #workers=1,
                             #use_multiprocessing=True  # Enable this if your generator supports multiprocessing
#                            )
    
    # Clear session after each epoch to prevent memory buildup
#    K.clear_session()
#cursor:

# 1. Setup memory management - THIS MUST BE AT THE VERY START OF YOUR SCRIPT
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

# Configure GPU - must be done before any other TensorFlow operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Just set memory growth, remove the memory limit configuration
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. Define callbacks
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path.replace('.h5', '.keras'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30, #was previously 10
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tensorboard
]

# 3. Training with safety measures
try:
    # Clear memory
    gc.collect()
    K.clear_session()
    
    # Start training - removed workers and use_multiprocessing parameters
    #loss_history = model.fit(
    #    train_generator,
    #    steps_per_epoch=config.MAX_TRAIN_STEPS,
    #    epochs=config.NB_EPOCHS,
    #    validation_data=val_generator,
    #    validation_steps=config.VALID_IMG_COUNT,
    #    callbacks=callbacks_list
    #)
    loss_history = model.fit(
        train_generator, 
        steps_per_epoch=config.MAX_TRAIN_STEPS,
        epochs=config.NB_EPOCHS,
        validation_data=val_generator,
        validation_steps=config.VALID_IMG_COUNT,
        callbacks=callbacks_list  # Remove class_weight parameter
    )
    
    # Save final model
    model.save(model_path.replace('.h5', '_final.keras'))

except Exception as e:
    print(f"Training failed with error: {str(e)}")
    try:
        model.save(model_path.replace('.h5', '_emergency_save.keras'))
    except:
        print("Could not save model state")
    
finally:
    gc.collect()
    K.clear_session()


# In[ ]:


# Load model after training
# If you load a model with different python version, than you may run into a problem: https://github.com/keras-team/keras/issues/9595#issue-303471777

#model = load_model(model_path, custom_objects={'tversky': LOSS, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy , 'specificity': specificity, 'sensitivity':sensitivity}, compile=False)

# In case you want to use multiple GPU you can uncomment the following lines.
# from tensorflow.python.keras.utils import multi_gpu_model
# model = multi_gpu_model(model, gpus=2, cpu_merge=False)

#model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])

#cursor:
import os
import glob

# Get all .keras files in the model directory
model_files = glob.glob(os.path.join(config.model_path, '*.keras'))
# Sort by modification time, newest first
latest_model = max(model_files, key=os.path.getmtime)
print(f"Latest model file: {latest_model}")

model = load_model(latest_model, 
                  custom_objects={
                      'tversky': LOSS, 
                      'dice_coef': dice_coef, 
                      'dice_loss': dice_loss, 
                      'accuracy': accuracy, 
                      'specificity': specificity, 
                      'sensitivity': sensitivity
                  }, 
                  compile=False)


# In[ ]:


# Print one batch on the training/test data!
for i in range(1):
    test_images, real_label = next(test_generator)
    #5 images per row: pan, ndvi, label, weight, prediction
    prediction = model.predict(test_images, steps=1)
    prediction[prediction>0.5]=1
    prediction[prediction<=0.5]=0
    display_images(np.concatenate((test_images, real_label, prediction), axis = -1))


# In[ ]:




