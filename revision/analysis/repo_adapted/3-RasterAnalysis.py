"""
Generated from notebooks/3-RasterAnalysis.ipynb
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import rasterio                  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.warp             # Reproject raster samples
from rasterio import windows
import fiona                     # I/O vector data (shape, geojson, ...)
import geopandas as gps

from shapely.geometry import Point, Polygon
from shapely.geometry import mapping, shape

import numpy as np               # numerical array manipulation
import os
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
import pandas as pd

from itertools import product
from tensorflow.keras.models import load_model

import sys

# Set up path for core module BEFORE importing it
# Base directory (scratch run) - must be set before core imports
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
# Ensure Python can import the local 'core' package when this script is run
# from outside the notebooks_scratch directory (e.g. as os.path.join(BASE_DIR, '3-RasterAnalysis.py'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)  # Insert at beginning for priority

from core.UNet import UNet
from core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
from core.optimizers import adaDelta, adagrad, adam, nadam
from core.frame_info import FrameInfo, image_normalize
from core.dataset_generator import DataGenerator
from core.split_frames import split_dataset
from core.visualize import display_images

# %matplotlib inline
# Set non-interactive backend for batch processing (no display required)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # plotting tools
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings                  # ignore annoying warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Additional imports needed for batch processing
import cv2
from collections import defaultdict

# %reload_ext autoreload
# %autoreload 2
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import tensorflow as tf
# Test GPU computation
#try:
#    with tf.device('/GPU:0'):
#        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#        c = tf.matmul(a, b)
#        print("Tensor computed on GPU:")
#        print(c)
#except RuntimeError as e:
 #   print("GPU error:", e)

## Configure TensorFlow for better performance
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        # Enable memory growth
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable XLA optimization
#        tf.config.optimizer.set_jit(True)
        
        # Optional: Set mixed precision
#        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
#        print("GPU configuration successful")
#    except RuntimeError as e:
#        print("GPU configuration error:", e)
#        print("Falling back to CPU")
#        # Disable GPU
#        tf.config.set_visible_devices([], 'GPU')
#else:
#    print("No GPU devices found. Using CPU.")

# Force CPU if GPU is unavailable
#try:
#    # Try to create a CPU-only session
#    tf.config.set_visible_devices([], 'GPU')
#    print("Successfully configured for CPU-only operation")
#except:
#    print("Error configuring devices")

# Required configurations (including the input and output paths) are stored in a separate file (such as config/RasterAnalysis.py)
# Please provide required info in the file before continuing with this notebook. 
 
from config import RasterAnalysis
# In case you are using a different folder name such as configLargeCluster, then you should import from the respective folder 
# Eg. from configLargeCluster import RasterAnalysis

config = RasterAnalysis.Configuration()

# Cutouts directory (e.g. myenv_scratch run: BASE_DIR/cutouts)
# BASE_DIR is already defined above (before core imports)
CUTOUTS_DIR = os.path.join(BASE_DIR, "cutouts")

# Load a pretrained model
#OPTIMIZER = adaDelta
#model = load_model(config.trained_model_path, custom_objects={'tversky': tversky, 'dice_coef': dice_coef, 'dice_loss':dice_loss, 'accuracy':accuracy, 'specificity':specificity, 'sensitivity':sensitivity}, compile=False)
#model.compile(optimizer=OPTIMIZER, loss=tversky, metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])


#cursor:

import glob
import os

# Load a pretrained model
OPTIMIZER = adaDelta

# Model directory path (use BASE_DIR for absolute path)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models", "UNet")

# Default model filename (if specific model is needed as fallback)
DEFAULT_MODEL_FILENAME = "trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_FILENAME)

print(f"Looking for models in: {MODEL_DIR}")

try:
    # First, try to use config.trained_model_path if it's an absolute path
    if os.path.isabs(config.trained_model_path):
        model_search_dir = config.trained_model_path
    else:
        # If relative, try relative to BASE_DIR first
        model_search_dir = os.path.join(BASE_DIR, config.trained_model_path)
        if not os.path.exists(model_search_dir):
            # Fallback to just the config path (might be relative to current dir)
            model_search_dir = config.trained_model_path
    
    # Get all .keras files in the model directory
    model_files = glob.glob(os.path.join(model_search_dir, '*.keras'))
    
    # If not found in config path, try the MODEL_DIR
    if not model_files:
        print(f"No .keras files found in {model_search_dir}, trying {MODEL_DIR}")
        model_files = glob.glob(os.path.join(MODEL_DIR, '*.keras'))
    
    if model_files:  # If files were found
        # Sort by modification time, newest first
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Loading latest model: {latest_model}")
    else:
        # Fallback to default model path
        if os.path.exists(DEFAULT_MODEL_PATH):
            latest_model = DEFAULT_MODEL_PATH
            print(f"No .keras files found in search directories. Using default model: {latest_model}")
        else:
            raise FileNotFoundError(
                f"No model files found!\n"
                f"Searched in: {model_search_dir}\n"
                f"Also searched in: {MODEL_DIR}\n"
                f"Default model not found: {DEFAULT_MODEL_PATH}\n"
                f"Please ensure model files exist in one of these locations."
            )
        
except Exception as e:
    # If there was an error, try the default path
    if os.path.exists(DEFAULT_MODEL_PATH):
        latest_model = DEFAULT_MODEL_PATH
        print(f"Error accessing model directory: {str(e)}. Using default model: {latest_model}")
    else:
        raise FileNotFoundError(
            f"Could not load model!\n"
            f"Error: {str(e)}\n"
            f"Default model not found: {DEFAULT_MODEL_PATH}\n"
            f"Please check that model files exist in: {MODEL_DIR}"
        )

model = load_model(latest_model, 
                  custom_objects={
                      'tversky': tversky, 
                      'dice_coef': dice_coef, 
                      'dice_loss': dice_loss, 
                      'accuracy': accuracy, 
                      'specificity': specificity, 
                      'sensitivity': sensitivity
                  }, 
                  compile=False)

model.compile(optimizer=OPTIMIZER, loss=tversky, 
             metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])

# Methods to add results of a patch to the total results of a larger area. The operator could be min (useful if there are too many false positives), max (useful for tackle false negatives)
def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions) 
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    else: #operator == 'REPLACE':
        resultant = newPredictions    
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get. 
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he, col:col+wi] =  resultant
    return (res)

# Methods that actually makes the predictions
#def predict_using_model(model, batch, batch_pos, mask, operator):
#    tm = np.stack(batch, axis = 0)
#    prediction = model.predict(tm)
#    for i in range(len(batch_pos)):
#        (col, row, wi, he) = batch_pos[i]
#        p = np.squeeze(prediction[i], axis = -1)
#        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
#        mask = addTOResult(mask, p, row, col, he, wi, operator)
#    return mask
    

#def detect_tree(ndvi_img, pan_img, width=256, height=256, stride = 128, normalize=True):
#    assert ndvi_img.meta['width'] == pan_img.meta['width'] and ndvi_img.meta['height'] == pan_img.meta['height']
#    nols, nrows = ndvi_img.meta['width'], ndvi_img.meta['height']
#    meta = ndvi_img.meta.copy()
#    if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
#        meta['dtype'] = np.float32
#    offsets = product(range(0, nols, stride), range(0, nrows, stride))
#    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
##     print(nrows, nols)

#    mask = np.zeros((nrows, nols), dtype=meta['dtype'])

##     mask = mask -1 # Note: The initial mask is initialized with -1 instead of zero to handle the MIN case (see addToResult)
#    batch = []
#    batch_pos = [ ]
#    for col_off, row_off in  tqdm(offsets):
#        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
#        transform = windows.transform(window, ndvi_img.transform)
#        patch = np.zeros((height, width, 2)) #Add zero padding in case of corner images
#        ndvi_sm = ndvi_img.read(window=window)
#        pan_sm = pan_img.read(window=window)
#        temp_im = np.stack((ndvi_sm, pan_sm), axis = -1)
#        temp_im = np.squeeze(temp_im)
        
#        if normalize:
#            temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
            
#        patch[:window.height, :window.width] = temp_im
#        batch.append(patch)
#        batch_pos.append((window.col_off, window.row_off, window.width, window.height))
#        if (len(batch) == config.BATCH_SIZE):
#            mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#            batch = []
#            batch_pos = []
            
#    # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
#    if batch:
#        mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#        batch = []
#        batch_pos = []

#    return(mask, meta)

#cursor

#def predict_using_model(model, batch, batch_pos, mask, operator):
#    try:
        # Reduce batch size if needed
#        SAFE_BATCH_SIZE = 4  # Smaller batch size to prevent memory issues
        
#        for i in range(0, len(batch), SAFE_BATCH_SIZE):
#            batch_slice = batch[i:i + SAFE_BATCH_SIZE]
#            pos_slice = batch_pos[i:i + SAFE_BATCH_SIZE]
            
            # Stack the smaller batch
#            tm = np.stack(batch_slice, axis=0)
            
            # Make prediction
#            prediction = model.predict(tm, verbose=0)  # Turn off verbose output
            
            # Add diagnostic print
#            print(f"\nGPU Batch prediction stats:")
#            print(f"Shape: {prediction.shape}, Range: [{prediction.min():.3f}, {prediction.max():.3f}], Mean: {prediction.mean():.3f}")
            
            # Process predictions
#            for j in range(len(pos_slice)):
#                (col, row, wi, he) = pos_slice[j]
#                p = np.squeeze(prediction[j], axis=-1)
#                mask = addTOResult(mask, p, row, col, he, wi, operator)
            
#    except Exception as e:
#        print(f"GPU prediction failed with error: {str(e)}")
#        print("Attempting to continue with CPU...")
        
        # Try processing one image at a time on CPU
#        with tf.device('/CPU:0'):
#            for single_img, single_pos in zip(batch, batch_pos):
#                tm = np.expand_dims(single_img, axis=0)
#                prediction = model.predict(tm, verbose=0)
                
                # Add diagnostic print
#                print(f"\nCPU Single prediction stats:")
#                print(f"Shape: {prediction.shape}, Range: [{prediction.min():.3f}, {prediction.max():.3f}], Mean: {prediction.mean():.3f}")
                
#                p = np.squeeze(prediction[0], axis=-1)
#                (col, row, wi, he) = single_pos
#                mask = addTOResult(mask, p, row, col, he, wi, operator)
                
#    return mask

#def detect_tree(ndvi_band, pan_band, width=256, height=256, stride=128, normalize=True):
    # Verify input arrays have the same shape
#    assert ndvi_band.shape == pan_band.shape, "NDVI and PAN bands must have the same dimensions"
    
#    nrows, nols = ndvi_band.shape
#    mask = np.zeros((nrows, nols), dtype=np.float32)

#    offsets = product(range(0, nols, stride), range(0, nrows, stride))
#    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    
#    batch = []
#    batch_pos = []
    
    # Reduce max batch size
#    MAX_BATCH_SIZE = 8  # Smaller than default to prevent memory issues

#    for col_off, row_off in tqdm(offsets):
#        actual_width = min(width, nols - col_off)
#        actual_height = min(height, nrows - row_off)
        
#        if actual_width > 0 and actual_height > 0:
#            window = windows.Window(col_off=col_off, row_off=row_off, 
#                                  width=actual_width, height=actual_height).intersection(big_window)
#            patch = np.zeros((height, width, 2))
            
#            ndvi_patch = ndvi_band[row_off:row_off + actual_height, 
#                                 col_off:col_off + actual_width]
#            pan_patch = pan_band[row_off:row_off + actual_height, 
#                               col_off:col_off + actual_width]
            
#            temp_im = np.stack((ndvi_patch, pan_patch), axis=-1)

#            if normalize:
#                temp_im = image_normalize(temp_im, axis=(0, 1))
            
#            patch[:actual_height, :actual_width, :] = temp_im
            
#            batch.append(patch)
#            batch_pos.append((col_off, row_off, actual_width, actual_height))

#            if len(batch) == MAX_BATCH_SIZE:
#                mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
#                batch = []
#                batch_pos = []

#    if batch:
#        mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')

#    meta = {
#        'width': nols,
#        'height': nrows,
#        'dtype': mask.dtype,
#        'count': 1  # Single band for the mask
#    }

#    return mask, meta

# new on 12/02
# Prediction and blob splitting functions for tree detection

def predict_using_model_with_confidence(model, batch, batch_pos, mask, operator, 
                                         confidence_map, overlap_count):
    """
    Make predictions using the model on a batch of patches and add to the mask.
    Also tracks confidence by accumulating prediction values and overlap counts.
    
    Returns:
        mask: Updated prediction mask
        confidence_map: Updated confidence map (sum of predictions)
        overlap_count: Updated overlap count map
    """
    try:
        SAFE_BATCH_SIZE = 4
        
        for i in range(0, len(batch), SAFE_BATCH_SIZE):
            batch_slice = batch[i:i + SAFE_BATCH_SIZE]
            pos_slice = batch_pos[i:i + SAFE_BATCH_SIZE]
            
            tm = np.stack(batch_slice, axis=0)
            prediction = model.predict(tm, verbose=0)
            
            for j in range(len(pos_slice)):
                (col, row, wi, he) = pos_slice[j]
                p = np.squeeze(prediction[j], axis=-1)
                
                # Update mask
                mask = addTOResult(mask, p, row, col, he, wi, operator)
                
                # Update confidence map (accumulate predictions)
                confidence_map[row:row+he, col:col+wi] += p[:he, :wi]
                overlap_count[row:row+he, col:col+wi] += 1
        
    except Exception as e:
        print(f"Prediction failed with error: {str(e)}")
        with tf.device('/CPU:0'):
            for single_img, single_pos in zip(batch, batch_pos):
                tm = np.expand_dims(single_img, axis=0)
                prediction = model.predict(tm, verbose=0)
                
                p = np.squeeze(prediction[0], axis=-1)
                (col, row, wi, he) = single_pos
                mask = addTOResult(mask, p, row, col, he, wi, operator)
                confidence_map[row:row+he, col:col+wi] += p[:he, :wi]
                overlap_count[row:row+he, col:col+wi] += 1
    
    return mask, confidence_map, overlap_count

def predict_using_model(model, batch, batch_pos, mask, operator):
    """
    Make predictions using the model on a batch of patches and add to the mask.
    """
    try:
        # Reduce batch size if needed
        SAFE_BATCH_SIZE = 4  # Smaller batch size to prevent memory issues
        
        for i in range(0, len(batch), SAFE_BATCH_SIZE):
            batch_slice = batch[i:i + SAFE_BATCH_SIZE]
            pos_slice = batch_pos[i:i + SAFE_BATCH_SIZE]
            
            # Stack the smaller batch
            tm = np.stack(batch_slice, axis=0)
            
            # Make prediction
            prediction = model.predict(tm, verbose=0)  # Turn off verbose output
            
            # Process predictions
            for j in range(len(pos_slice)):
                (col, row, wi, he) = pos_slice[j]
                p = np.squeeze(prediction[j], axis=-1)
                mask = addTOResult(mask, p, row, col, he, wi, operator)
            
    except Exception as e:
        print(f"GPU prediction failed with error: {str(e)}")
        print("Attempting to continue with CPU...")
        
        # Try processing one image at a time on CPU
        with tf.device('/CPU:0'):
            for single_img, single_pos in zip(batch, batch_pos):
                tm = np.expand_dims(single_img, axis=0)
                prediction = model.predict(tm, verbose=0)
                
                p = np.squeeze(prediction[0], axis=-1)
                (col, row, wi, he) = single_pos
                mask = addTOResult(mask, p, row, col, he, wi, operator)
                
    return mask

def split_large_tree_blobs(mask, max_tree_area=5000, min_tree_area=50, 
                           threshold=0.5, min_distance=10):
    """
    Split large tree blobs into individual trees based on expected tree size.
    
    This function identifies connected components in the prediction mask and splits
    any blob that is larger than max_tree_area into multiple smaller trees using
    watershed segmentation based on distance transform.
    
    Args:
        mask: Prediction mask (2D float array, 0-1 range)
        max_tree_area: Maximum area (in pixels) for a single tree. Blobs larger than
                      this will be split into multiple trees. Default: 5000 pixels
        min_tree_area: Minimum area (in pixels) for a valid tree. Smaller blobs will
                      be removed. Default: 50 pixels
        threshold: Threshold for converting prediction mask to binary. Default: 0.5
        min_distance: Minimum distance (in pixels) between tree centers when splitting.
                     Default: 10 pixels
    
    Returns:
        Processed mask with large blobs split into individual trees
    """
    from skimage import measure, segmentation
    from scipy import ndimage
    
    # Helper function to find local maxima using scipy (more compatible)
    def find_local_maxima(distance, min_distance, threshold_abs):
        """Find local maxima in distance transform using scipy.ndimage"""
        # Use maximum filter to find local maxima
        # For 2D arrays, size should be a tuple or integer (applied to both dimensions)
        local_maxima = ndimage.maximum_filter(distance, size=min_distance, mode='constant')
        maxima_mask = (distance == local_maxima) & (distance >= threshold_abs)
        
        # Get coordinates of maxima
        y_coords, x_coords = np.where(maxima_mask)
        
        # Filter maxima to ensure minimum distance between them
        if len(y_coords) > 1:
            # Sort by distance value (highest first) and filter by minimum distance
            sorted_indices = np.argsort(distance[y_coords, x_coords])[::-1]
            filtered_y = []
            filtered_x = []
            
            for idx in sorted_indices:
                y, x = y_coords[idx], x_coords[idx]
                # Check if this point is far enough from existing points
                is_far_enough = True
                for fy, fx in zip(filtered_y, filtered_x):
                    dist = np.sqrt((y - fy)**2 + (x - fx)**2)
                    if dist < min_distance:
                        is_far_enough = False
                        break
                if is_far_enough:
                    filtered_y.append(y)
                    filtered_x.append(x)
            
            return (np.array(filtered_y), np.array(filtered_x))
        
        return (y_coords, x_coords)
    
    # Convert to binary mask
    binary_mask = (mask >= threshold).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return mask  # No trees detected
    
    # Label connected components
    labeled_mask, num_components = measure.label(binary_mask, return_num=True, connectivity=2)
    
    # Get properties of each component
    props = measure.regionprops(labeled_mask)
    
    # Create output mask
    output_mask = np.zeros_like(mask, dtype=np.float32)
    
    print(f"Found {num_components} connected components")
    
    for prop in props:
        area = prop.area
        
        # Remove very small blobs (likely noise)
        if area < min_tree_area:
            continue
        
        # Get the bounding box and extract the blob
        min_row, min_col, max_row, max_col = prop.bbox
        blob_region = labeled_mask[min_row:max_row, min_col:max_col] == prop.label
        
        # Extract corresponding region from original mask (preserve prediction confidence)
        blob_mask = mask[min_row:max_row, min_col:max_col].copy()
        blob_mask[~blob_region] = 0
        
        # If blob is smaller than max_tree_area, keep it as is
        if area <= max_tree_area:
            output_mask[min_row:max_row, min_col:max_col] = np.maximum(
                output_mask[min_row:max_row, min_col:max_col],
                blob_mask
            )
        else:
            # Split large blob using watershed segmentation
            # Create binary mask for this blob
            blob_binary = blob_region.astype(np.uint8)
            
            # Compute distance transform
            distance = ndimage.distance_transform_edt(blob_binary)
            
            # Find local maxima (potential tree centers) using our helper function
            threshold_value = distance.max() * 0.3  # Only consider peaks at least 30% of max distance
            y_coords, x_coords = find_local_maxima(distance, min_distance, threshold_value)
            
            # If we found multiple peaks, use watershed to split
            if len(y_coords) > 1:
                # Create markers for watershed
                markers = np.zeros_like(blob_binary, dtype=np.int32)
                for i, (y, x) in enumerate(zip(y_coords, x_coords), start=1):
                    markers[y, x] = i
                
                # Apply watershed segmentation
                # Use inverted distance as the image (watershed fills from markers)
                labels = segmentation.watershed(
                    -distance,  # Invert distance so peaks become valleys
                    markers,
                    mask=blob_binary
                )
                
                # Transfer segmented regions back to output mask
                # Use the original prediction values, not just binary
                for label_id in range(1, labels.max() + 1):
                    label_mask = (labels == label_id)
                    # Preserve original prediction confidence values
                    label_values = blob_mask.copy()
                    label_values[~label_mask] = 0
                    output_mask[min_row:max_row, min_col:max_col] = np.maximum(
                        output_mask[min_row:max_row, min_col:max_col],
                        label_values
                    )
            else:
                # Only one peak found, keep the blob as is (but it's still too large)
                # This might be a genuinely large tree, or we couldn't split it
                output_mask[min_row:max_row, min_col:max_col] = np.maximum(
                    output_mask[min_row:max_row, min_col:max_col],
                    blob_mask
                )
    
    return output_mask

def analyze_blob_sizes(mask, threshold=0.5):
    """
    Analyze the sizes of connected components (blobs) in the prediction mask.
    Useful for determining appropriate max_tree_area and min_tree_area parameters.
    
    Args:
        mask: Prediction mask (2D float array)
        threshold: Threshold for converting to binary mask (default: 0.5)
    
    Returns:
        Dictionary with blob size statistics
    """
    from skimage import measure
    
    # Convert to binary
    binary_mask = (mask >= threshold).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return {'num_blobs': 0, 'message': 'No trees detected in mask'}
    
    # Label connected components
    labeled_mask, num_components = measure.label(binary_mask, return_num=True, connectivity=2)
    props = measure.regionprops(labeled_mask)
    
    # Calculate statistics
    areas = [prop.area for prop in props]
    
    if len(areas) == 0:
        return {'num_blobs': 0, 'message': 'No valid blobs found'}
    
    areas = np.array(areas)
    
    stats = {
        'num_blobs': len(areas),
        'total_area': int(np.sum(areas)),
        'mean_area': float(np.mean(areas)),
        'median_area': float(np.median(areas)),
        'std_area': float(np.std(areas)),
        'min_area': int(np.min(areas)),
        'max_area': int(np.max(areas)),
        'percentile_25': float(np.percentile(areas, 25)),
        'percentile_75': float(np.percentile(areas, 75)),
        'percentile_90': float(np.percentile(areas, 90)),
        'percentile_95': float(np.percentile(areas, 95)),
        'percentile_99': float(np.percentile(areas, 99)),
        'large_blobs_count': int(np.sum(areas > 5000)),  # Count blobs > 5000 pixels
        'large_blobs_percentage': float(100 * np.sum(areas > 5000) / len(areas)),
    }
    
    # Print summary
    print("=" * 60)
    print("Blob Size Analysis")
    print("=" * 60)
    print(f"Total number of blobs: {stats['num_blobs']}")
    print(f"Total area (pixels): {stats['total_area']:,}")
    print(f"\nArea Statistics (pixels):")
    print(f"  Mean: {stats['mean_area']:.1f}")
    print(f"  Median: {stats['median_area']:.1f}")
    print(f"  Std Dev: {stats['std_area']:.1f}")
    print(f"  Min: {stats['min_area']}")
    print(f"  Max: {stats['max_area']:,}")
    print(f"\nPercentiles:")
    print(f"  25th: {stats['percentile_25']:.1f}")
    print(f"  75th: {stats['percentile_75']:.1f}")
    print(f"  90th: {stats['percentile_90']:.1f}")
    print(f"  95th: {stats['percentile_95']:.1f}")
    print(f"  99th: {stats['percentile_99']:.1f}")
    print(f"\nLarge Blobs (>5000 pixels):")
    print(f"  Count: {stats['large_blobs_count']}")
    print(f"  Percentage: {stats['large_blobs_percentage']:.1f}%")
    print("=" * 60)
    
    # Suggest parameters
    print("\nSuggested Parameters:")
    print(f"  min_tree_area: {max(50, int(stats['percentile_25'] * 0.5))}  (to filter noise)")
    print(f"  max_tree_area: {int(stats['percentile_90'])}  (to split large blobs)")
    print("=" * 60)
    
    return stats

schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'str', 'canopy': 'float:15.2',},
    }

def drawPolygons(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=255, fill=255)
    mask = np.array(mask)#, dtype=bool)   
    return(mask)

def transformToXY(polygons, transform):
    tp = []
    for polygon in polygons:
        rows, cols = zip(*polygon)
        x,y = rasterio.transform.xy(transform, rows, cols)
        tp.append(list(zip(x,y)))
    return (tp)

def createShapefileObject(polygons, meta, wfile):
    with fiona.open(wfile, 'w', crs=meta.get('crs').to_dict(), driver='ESRI Shapefile', schema=schema) as sink:
        for idx, mp in enumerate(polygons):
            try:
#                 poly = Polygon(poly)
    #             assert mp.is_valid
    #             assert mp.geom_type == 'Polygon'
                sink.write({
                    'geometry': mapping(mp),
                    'properties': {'id': str(idx), 'canopy': mp.area},
                })
            except:
                print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
#                 print(mp)

# Generate a mask with polygons
def transformContoursToXY(contours, transform = None):
    tp = []
    for cnt in contours:
        pl = cnt[:, 0, :]
        cols, rows = zip(*pl)
        x,y = rasterio.transform.xy(transform, rows, cols)
        tl = [list(i) for i in zip(x, y)]
        tp.append(tl)
    return (tp)


def mask_to_polygons(maskF, transform):
    # first, find contours with cv2: it's much faster than shapely
    th = 0.7  # Updated threshold
    mask = maskF.copy()
    mask[mask < th] = 0
    mask[mask >= th] = 1
    mask = ((mask) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    #Convert contours from image coordinate to xy coordinate
    contours = transformContoursToXY(contours, transform)
    if not contours: #TODO: Raise an error maybe
        print('Warning: No contours/polygons detected!!')
        return [Polygon()]
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []

    for idx, cnt in enumerate(contours):
        if idx not in child_contours: #and cv2.contourArea(cnt) >= min_area: #Do we need to check for min_area??
            try:
                poly = Polygon(
                    shell=cnt,
                    holes=[c for c in cnt_children.get(idx, [])])
                           #if cv2.contourArea(c) >= min_area]) #Do we need to check for min_area??
                all_polygons.append(poly)
            except:
                pass
#                 print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
    print(len(all_polygons))
    return(all_polygons)

def create_contours_shapefile(mask, meta, out_fn):
    res = mask_to_polygons(mask, meta['transform'])
#     res = transformToXY(contours, meta['transform'])
    createShapefileObject(res, meta, out_fn)


def writeMaskToDisk(detected_mask, detected_meta, wp, write_as_type = 'uint8', th = 0.7, create_countors = False):
    # Convert to correct required before writing
    if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
        print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
        detected_mask[detected_mask<th]=0
        detected_mask[detected_mask>=th]=1
        detected_mask = detected_mask.astype(write_as_type)
        detected_meta['dtype'] =  write_as_type
        
    with rasterio.open(wp, 'w', **detected_meta) as outds:
        outds.write(detected_mask, 1)
    if create_countors:
        # Replace file extension with .shp for shapefile output
        wp_shp = os.path.splitext(wp)[0] + '.shp'
        create_contours_shapefile(detected_mask, detected_meta, wp_shp)

# detect_tree function for tree detection using UNet model
def detect_tree(ndvi_band, pan_band, width=256, height=256, stride=128, normalize=True, return_confidence=False):
    """
    Detect trees in NDVI and PAN bands using UNet model.
    
    Args:
        ndvi_band: NDVI band as numpy array
        pan_band: PAN band as numpy array
        width: Width of prediction patches (default: 256)
        height: Height of prediction patches (default: 256)
        stride: Stride for sliding window (default: 128)
        normalize: Whether to normalize patches (default: True)
        return_confidence: Whether to return confidence map (default: False)
    
    Returns:
        If return_confidence=False:
            mask: Prediction mask (2D float array)
            meta: Metadata dictionary
        If return_confidence=True:
            mask: Prediction mask (2D float array)
            meta: Metadata dictionary
            confidence_map: Confidence map (average of overlapping predictions)
    """
    # Verify input arrays have the same shape
    assert ndvi_band.shape == pan_band.shape, "NDVI and PAN bands must have the same dimensions"
    
    nrows, nols = ndvi_band.shape
    mask = np.zeros((nrows, nols), dtype=np.float32)
    
    # Initialize confidence map if needed
    if return_confidence:
        confidence_map = np.zeros((nrows, nols), dtype=np.float32)
        overlap_count = np.zeros((nrows, nols), dtype=np.int32)
    
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    
    batch = []
    batch_pos = []
    
    # Reduce max batch size to prevent memory issues
    MAX_BATCH_SIZE = 8
    
    for col_off, row_off in tqdm(offsets, desc="Processing patches"):
        actual_width = min(width, nols - col_off)
        actual_height = min(height, nrows - row_off)
        
        if actual_width > 0 and actual_height > 0:
            window = windows.Window(col_off=col_off, row_off=row_off, 
                                  width=actual_width, height=actual_height).intersection(big_window)
            patch = np.zeros((height, width, 2))
            
            ndvi_patch = ndvi_band[row_off:row_off + actual_height, 
                                 col_off:col_off + actual_width]
            pan_patch = pan_band[row_off:row_off + actual_height, 
                               col_off:col_off + actual_width]
            
            temp_im = np.stack((ndvi_patch, pan_patch), axis=-1)
            
            if normalize:
                temp_im = image_normalize(temp_im, axis=(0, 1))
            
            patch[:actual_height, :actual_width, :] = temp_im
            
            batch.append(patch)
            batch_pos.append((col_off, row_off, actual_width, actual_height))
            
            if len(batch) == MAX_BATCH_SIZE:
                if return_confidence:
                    mask, confidence_map, overlap_count = predict_using_model_with_confidence(
                        model, batch, batch_pos, mask, 'MAX', confidence_map, overlap_count)
                else:
                    mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
                batch = []
                batch_pos = []
    
    # Process remaining batch
    if batch:
        if return_confidence:
            mask, confidence_map, overlap_count = predict_using_model_with_confidence(
                model, batch, batch_pos, mask, 'MAX', confidence_map, overlap_count)
        else:
            mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
    
    # Create metadata
    meta = {
        'width': nols,
        'height': nrows,
        'dtype': mask.dtype,
        'count': 1  # Single band for the mask
    }
    
    if return_confidence:
        # Normalize confidence map by overlap count
        with np.errstate(divide='ignore', invalid='ignore'):
            confidence_map = np.divide(confidence_map, overlap_count, 
                                     out=np.zeros_like(confidence_map), 
                                     where=overlap_count > 0)
        return mask, meta, confidence_map
    else:
        return mask, meta

#new on 12/02
# Functions for city-center area identification and analysis filtering

def identify_city_center_areas(pan_band, method='density_percentile', percentile=75, 
                                kernel_size=100, min_density_threshold=None):
    """
    Identify city-center areas based on urban density from PAN band.
    
    City-center areas are typically characterized by higher building density,
    which manifests as higher variance and intensity in the PAN band.
    
    Args:
        pan_band: PAN band as numpy array
        method: Method to identify city-center ('density_percentile', 'variance', 'intensity')
        percentile: Percentile threshold for density-based method (default: 75)
        kernel_size: Size of kernel for density calculation (default: 100 pixels)
        min_density_threshold: Optional absolute threshold for density (overrides percentile)
    
    Returns:
        city_center_mask: Boolean mask where True indicates city-center areas
        urban_density_map: Map of urban density values (0-1 scale)
    """
    from scipy import ndimage
    
    # Normalize PAN band to 0-1 range for consistent processing
    pan_normalized = (pan_band - pan_band.min()) / (pan_band.max() - pan_band.min() + 1e-10)
    
    if method == 'density_percentile':
        # Calculate local density using variance (urban areas have higher variance)
        # Use a moving window to calculate local variance
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = ndimage.convolve(pan_normalized, kernel, mode='constant')
        local_variance = ndimage.convolve((pan_normalized - local_mean)**2, kernel, mode='constant')
        
        # Combine intensity and variance as urban density indicator
        urban_density_map = 0.5 * pan_normalized + 0.5 * (local_variance / (local_variance.max() + 1e-10))
        
        # Threshold based on percentile
        if min_density_threshold is None:
            threshold = np.percentile(urban_density_map, percentile)
        else:
            threshold = min_density_threshold
        
        city_center_mask = urban_density_map >= threshold
        
    elif method == 'variance':
        # Use variance as indicator
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = ndimage.convolve(pan_normalized, kernel, mode='constant')
        local_variance = ndimage.convolve((pan_normalized - local_mean)**2, kernel, mode='constant')
        urban_density_map = local_variance / (local_variance.max() + 1e-10)
        
        if min_density_threshold is None:
            threshold = np.percentile(urban_density_map, percentile)
        else:
            threshold = min_density_threshold
        
        city_center_mask = urban_density_map >= threshold
        
    elif method == 'intensity':
        # Use intensity as indicator
        urban_density_map = pan_normalized
        
        if min_density_threshold is None:
            threshold = np.percentile(urban_density_map, percentile)
        else:
            threshold = min_density_threshold
        
        city_center_mask = urban_density_map >= threshold
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'density_percentile', 'variance', or 'intensity'")
    
    return city_center_mask, urban_density_map


def filter_analysis_to_city_center(mask, city_center_mask, urban_density_map=None):
    """
    Filter analysis results to only city-center areas.
    
    Args:
        mask: Prediction mask (2D array)
        city_center_mask: Boolean mask indicating city-center areas
        urban_density_map: Optional urban density map for stratification
    
    Returns:
        filtered_mask: Mask filtered to city-center areas only
        stats: Dictionary with statistics about filtered vs. full area
    """
    filtered_mask = mask.copy()
    filtered_mask[~city_center_mask] = 0
    
    # Calculate statistics
    total_pixels = mask.size
    city_center_pixels = np.sum(city_center_mask)
    city_center_fraction = city_center_pixels / total_pixels
    
    # Tree detection statistics
    full_area_trees = np.sum(mask > 0.5)
    city_center_trees = np.sum(filtered_mask > 0.5)
    
    stats = {
        'total_pixels': int(total_pixels),
        'city_center_pixels': int(city_center_pixels),
        'city_center_fraction': float(city_center_fraction),
        'full_area_tree_pixels': int(full_area_trees),
        'city_center_tree_pixels': int(city_center_trees),
        'full_area_tree_coverage': float(full_area_trees / total_pixels),
        'city_center_tree_coverage': float(city_center_trees / city_center_pixels) if city_center_pixels > 0 else 0.0,
    }
    
    return filtered_mask, stats


def calculate_stratified_metrics(mask, urban_density_map, n_strata=5, threshold=0.5):
    """
    Calculate prediction metrics stratified by urban density.
    
    Args:
        mask: Prediction mask (2D array)
        urban_density_map: Urban density map (0-1 scale)
        n_strata: Number of density strata to create (default: 5)
        threshold: Threshold for binary tree classification (default: 0.5)
    
    Returns:
        stratified_stats: Dictionary with metrics for each density stratum
    """
    binary_mask = (mask >= threshold).astype(bool)
    
    # Create density strata
    density_bins = np.linspace(0, 1, n_strata + 1)
    strata_labels = []
    strata_stats = []
    
    for i in range(n_strata):
        lower_bound = density_bins[i]
        upper_bound = density_bins[i + 1]
        
        # Create mask for this stratum
        if i == n_strata - 1:  # Include upper bound for last stratum
            stratum_mask = (urban_density_map >= lower_bound) & (urban_density_map <= upper_bound)
        else:
            stratum_mask = (urban_density_map >= lower_bound) & (urban_density_map < upper_bound)
        
        stratum_pixels = np.sum(stratum_mask)
        
        if stratum_pixels > 0:
            stratum_tree_pixels = np.sum(binary_mask & stratum_mask)
            stratum_tree_coverage = stratum_tree_pixels / stratum_pixels
            stratum_mean_density = np.mean(urban_density_map[stratum_mask])
            stratum_mean_confidence = np.mean(mask[stratum_mask]) if np.any(stratum_mask) else 0.0
        else:
            stratum_tree_pixels = 0
            stratum_tree_coverage = 0.0
            stratum_mean_density = (lower_bound + upper_bound) / 2
            stratum_mean_confidence = 0.0
        
        strata_labels.append(f"Density {i+1} ({lower_bound:.2f}-{upper_bound:.2f})")
        strata_stats.append({
            'density_range': (lower_bound, upper_bound),
            'pixels': int(stratum_pixels),
            'tree_pixels': int(stratum_tree_pixels),
            'tree_coverage': float(stratum_tree_coverage),
            'mean_urban_density': float(stratum_mean_density),
            'mean_prediction_confidence': float(stratum_mean_confidence),
        })
    
    return {
        'n_strata': n_strata,
        'strata': strata_stats,
        'strata_labels': strata_labels
    }


def print_model_coverage_transparency():
    """
    Print transparency statement about model coverage and limitations.
    """
    print("=" * 80)
    print("MODEL COVERAGE AND TRANSPARENCY STATEMENT")
    print("=" * 80)
    print()
    print("Model Training and Coverage:")
    print("  • This UNet model was trained on metro-wide areas covering diverse")
    print("    urban landscapes including city centers, suburbs, and peri-urban zones.")
    print("  • The model can generate predictions across the full metropolitan area.")
    print()
    print("Analysis Scope and Reliability:")
    print("  • Formal analysis and reporting are limited to city-center areas where")
    print("    predictions are most reliable.")
    print("  • City-center areas are identified based on urban density metrics")
    print("    derived from the PAN band (building density indicators).")
    print("  • Predictions in suburban and peri-urban areas may have lower reliability")
    print("    due to differences in training data distribution and landscape characteristics.")
    print()
    print("Confidence and Stratification:")
    print("  • Prediction confidence maps are available to identify areas with")
    print("    higher vs. lower prediction certainty.")
    print("  • Metrics are stratified by urban density to help readers understand")
    print("    where predictions are strong vs. weak.")
    print("  • Users should interpret results with awareness of these spatial variations.")
    print()
    print("Recommendations:")
    print("  • For policy and planning decisions, prioritize city-center results.")
    print("  • Use confidence maps to identify areas requiring validation.")
    print("  • Consider stratified metrics when comparing across different urban zones.")
    print("=" * 80)
    print()

# Example: Using city-center filtering and transparency statements
# This cell demonstrates how to incorporate city-center analysis into your workflow

# Print transparency statement
print_model_coverage_transparency()

# Example usage - only run if required variables are available
# Set RUN_EXAMPLE = True to run the example, or False to skip it
RUN_EXAMPLE = False  # Set to True when you have pan_band and detectedMask ready

if RUN_EXAMPLE:
    # Try to load pan_band if not already available
    if 'pan_band' not in locals():
        pan_file_path = None
        
        # Option 1: Try to use file_pairs if available
        if 'file_pairs' in locals() and 'cutouts_dir' in locals() and len(file_pairs) > 0:
            pan_file_path = os.path.join(cutouts_dir, file_pairs[0][0])
            print(f"Using PAN file from file_pairs: {pan_file_path}")
        
        # Option 2: Try to find PAN files in cutouts_dir
        elif 'cutouts_dir' in locals() and os.path.exists(cutouts_dir):
            pan_files = [f for f in os.listdir(cutouts_dir) 
                        if f.startswith('pan_') and f.endswith('.tif')]
            if pan_files:
                pan_file_path = os.path.join(cutouts_dir, pan_files[0])
                print(f"Using first PAN file found: {pan_file_path}")
        
        # Option 3: Try default cutouts directory
        elif os.path.exists("cutouts"):
            pan_files = [f for f in os.listdir("cutouts") 
                        if f.startswith('pan_') and f.endswith('.tif')]
            if pan_files:
                pan_file_path = os.path.join("cutouts", pan_files[0])
                print(f"Using first PAN file from default cutouts directory: {pan_file_path}")
        
        if pan_file_path and os.path.exists(pan_file_path):
            with rasterio.open(pan_file_path) as pan_src:
                pan_band = pan_src.read(1)
        else:
            print("Warning: Could not find PAN file. Please set pan_file_path manually or ensure cutouts_dir contains PAN files.")
            print("Example: pan_file_path = 'cutouts/pan_your_file.tif'")
            RUN_EXAMPLE = False
    
    # Check that detectedMask exists
    if 'detectedMask' not in locals():
        print("Warning: detectedMask not found. Please run detect_tree() first to generate the prediction mask.")
        RUN_EXAMPLE = False
    
    # Run the example if all requirements are met
    if RUN_EXAMPLE:

        # Identify city-center areas
        city_center_mask, urban_density_map = identify_city_center_areas(
            pan_band, 
            method='density_percentile', 
            percentile=75,  # Top 25% most urban areas
            kernel_size=100
        )
        
        # Filter predictions to city-center only
        filtered_mask, stats = filter_analysis_to_city_center(
            detectedMask, 
            city_center_mask, 
            urban_density_map
        )
        
        # Optional: Calculate stratified metrics by urban density
        stratified_metrics = calculate_stratified_metrics(
            detectedMask, 
            urban_density_map, 
            n_strata=5
        )
        
        # Print statistics
        print("\nCity-Center Filtering Statistics:")
        print(f"  City-center area: {stats['city_center_fraction']*100:.1f}% of total")
        print(f"  Tree coverage (full area): {stats['full_area_tree_coverage']*100:.2f}%")
        print(f"  Tree coverage (city-center): {stats['city_center_tree_coverage']*100:.2f}%")
        print("\nStratified Metrics by Urban Density:")
        for i, (label, stat) in enumerate(zip(stratified_metrics['strata_labels'], 
                                               stratified_metrics['strata'])):
            print(f"\n{label}:")
            print(f"  Pixels: {stat['pixels']:,}")
            print(f"  Tree coverage: {stat['tree_coverage']*100:.2f}%")
            print(f"  Mean prediction confidence: {stat['mean_prediction_confidence']:.3f}")
else:
    print("Example code skipped. Set RUN_EXAMPLE = True and ensure pan_band and detectedMask are available to run the example.")
    
# Visualization functions for confidence maps and city-center analysis

def visualize_city_center_analysis(pan_band, mask, city_center_mask, urban_density_map, 
                                   confidence_map=None, figsize=(15, 10)):
    """
    Create a comprehensive visualization showing city-center analysis.
    
    Args:
        pan_band: PAN band image
        mask: Prediction mask
        city_center_mask: Boolean mask for city-center areas
        urban_density_map: Urban density map
        confidence_map: Optional confidence map
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 3 if confidence_map is not None else 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. PAN band
    ax = axes[0]
    im = ax.imshow(pan_band, cmap='gray')
    ax.set_title('PAN Band (Input)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # 2. Urban density map
    ax = axes[1]
    im = ax.imshow(urban_density_map, cmap='hot')
    ax.set_title('Urban Density Map')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Urban Density')
    
    # 3. City-center mask overlay
    ax = axes[2]
    ax.imshow(pan_band, cmap='gray', alpha=0.7)
    overlay = city_center_mask.astype(float)
    overlay[~city_center_mask] = np.nan
    im = ax.imshow(overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('City-Center Areas (Red Overlay)')
    ax.axis('off')
    
    # 4. Full prediction mask
    ax = axes[3]
    im = ax.imshow(mask, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Full Area Predictions')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Prediction Confidence')
    
    # 5. City-center filtered predictions
    filtered_mask = mask.copy()
    filtered_mask[~city_center_mask] = 0
    ax = axes[4]
    im = ax.imshow(filtered_mask, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('City-Center Filtered Predictions')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Prediction Confidence')
    
    # 6. Confidence map (if provided)
    if confidence_map is not None:
        ax = axes[5]
        im = ax.imshow(confidence_map, cmap='plasma')
        ax.set_title('Prediction Confidence Map')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Confidence')
    
    plt.tight_layout()
    return fig


def create_city_center_report(mask, city_center_mask, urban_density_map, 
                               confidence_map=None, threshold=0.5):
    """
    Create a comprehensive report on city-center vs. full area analysis.
    
    Args:
        mask: Prediction mask
        city_center_mask: Boolean mask for city-center areas
        urban_density_map: Urban density map
        confidence_map: Optional confidence map
        threshold: Threshold for binary classification
    
    Returns:
        report: Dictionary with comprehensive statistics
    """
    # Filter to city-center
    filtered_mask, filter_stats = filter_analysis_to_city_center(
        mask, city_center_mask, urban_density_map
    )
    
    # Calculate stratified metrics
    stratified_metrics = calculate_stratified_metrics(
        mask, urban_density_map, n_strata=5, threshold=threshold
    )
    
    # Calculate confidence statistics if available
    confidence_stats = {}
    if confidence_map is not None:
        confidence_stats = {
            'mean_confidence_full': float(np.mean(confidence_map)),
            'mean_confidence_city_center': float(np.mean(confidence_map[city_center_mask])),
            'std_confidence_full': float(np.std(confidence_map)),
            'std_confidence_city_center': float(np.std(confidence_map[city_center_mask])),
            'min_confidence': float(np.min(confidence_map)),
            'max_confidence': float(np.max(confidence_map)),
        }
    
    report = {
        'filtering_stats': filter_stats,
        'stratified_metrics': stratified_metrics,
        'confidence_stats': confidence_stats if confidence_map is not None else None,
    }
    
    # Print report
    print("=" * 80)
    print("CITY-CENTER ANALYSIS REPORT")
    print("=" * 80)
    print("\n1. AREA COVERAGE:")
    print(f"   Total pixels: {filter_stats['total_pixels']:,}")
    print(f"   City-center pixels: {filter_stats['city_center_pixels']:,} "
          f"({filter_stats['city_center_fraction']*100:.1f}% of total)")
    
    print("\n2. TREE DETECTION:")
    print(f"   Full area tree pixels: {filter_stats['full_area_tree_pixels']:,}")
    print(f"   Full area tree coverage: {filter_stats['full_area_tree_coverage']*100:.2f}%")
    print(f"   City-center tree pixels: {filter_stats['city_center_tree_pixels']:,}")
    print(f"   City-center tree coverage: {filter_stats['city_center_tree_coverage']*100:.2f}%")
    
    if confidence_map is not None:
        print("\n3. PREDICTION CONFIDENCE:")
        print(f"   Mean confidence (full area): {confidence_stats['mean_confidence_full']:.3f}")
        print(f"   Mean confidence (city-center): {confidence_stats['mean_confidence_city_center']:.3f}")
        print(f"   Std confidence (full area): {confidence_stats['std_confidence_full']:.3f}")
        print(f"   Std confidence (city-center): {confidence_stats['std_confidence_city_center']:.3f}")
    
    print("\n4. STRATIFIED METRICS BY URBAN DENSITY:")
    for label, stat in zip(stratified_metrics['strata_labels'], 
                          stratified_metrics['strata']):
        print(f"\n   {label}:")
        print(f"     Pixels: {stat['pixels']:,}")
        print(f"     Tree coverage: {stat['tree_coverage']*100:.2f}%")
        print(f"     Mean prediction confidence: {stat['mean_prediction_confidence']:.3f}")
    
    print("\n" + "=" * 80)
    
    return report
# Processing loop to generate confidence maps for each city
# Note: Confidence maps are generated DURING prediction (not from existing predictions)
# The model must run to generate confidence maps, which track prediction agreement across overlapping patches
RUN_PROCESSING_LOOP_EXAMPLE = False  # Set to True to run the processing loop

# Set cutouts directory (default to "cutouts" if not already defined)
if 'cutouts_dir' not in locals():
    cutouts_dir = CUTOUTS_DIR

# Processing loop: Generate confidence maps (predictions are generated internally but not saved)
if RUN_PROCESSING_LOOP_EXAMPLE:
    # Auto-create file_pairs from cutouts directory if not already defined
    if 'file_pairs' not in locals() or not file_pairs:
        print(f"Creating file_pairs from {cutouts_dir} directory...")
        if not os.path.exists(cutouts_dir):
            print(f"Error: Directory '{cutouts_dir}' does not exist!")
        else:
            file_pairs = []
            # Get all PAN files
            all_files = os.listdir(cutouts_dir)
            pan_files = [f for f in all_files if f.startswith('pan_') and f.endswith('.tif') and 'atlanta' not in f.lower()]
            
            for pan_file in pan_files:
                # Find corresponding NDVI file (replace 'pan_' with 'ndvi_')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
                pan_path = os.path.join(cutouts_dir, pan_file)
                ndvi_path = os.path.join(cutouts_dir, ndvi_file)
                
                if os.path.exists(ndvi_path):
                    file_pairs.append((pan_path, ndvi_path))
            
            print(f"Found {len(file_pairs)} file pairs in {cutouts_dir}")
            if len(file_pairs) == 0:
                print("Warning: No matching PAN/NDVI file pairs found!")
                print(f"PAN files found: {len(pan_files)}")
    
    if 'file_pairs' in locals() and len(file_pairs) > 0:
        for pan_file, ndvi_file in file_pairs:
            # Extract filename from full path for output file naming
            pan_filename = os.path.basename(pan_file)
            predictionOutputFile = os.path.join(cutouts_dir, f"pred_{pan_filename}")
            
            # Check if confidence map already exists (skip if already processed)
            base_name = os.path.splitext(predictionOutputFile)[0]
            confidence_output_file = base_name + '_confidence.tif'
            if os.path.exists(confidence_output_file) and not getattr(config, 'overwrite_analysed_files', False):
                print(f"Skipping {pan_filename} - confidence map already exists")
                continue
            
            print(f"\nProcessing: {pan_filename}")
            
            with rasterio.open(pan_file) as pan_img, \
                 rasterio.open(ndvi_file) as ndvi_img:
                
                pan_band = pan_img.read(1)
                ndvi_band = ndvi_img.read(1)
                
                # Generate confidence map
                # Note: detect_tree() generates both predictions and confidence maps internally.
                # Confidence maps track how many overlapping patches agree on each pixel.
                # We only save the confidence map, not the predictions.
                detectedMask, detectedMeta, confidenceMap = detect_tree(
                    ndvi_band, pan_band,
                    width=config.WIDTH,
                    height=config.HEIGHT,
                    stride=config.STRIDE,
                    return_confidence=True
                )
                
                # Write prediction mask to disk
                writeMaskToDisk(
                    detectedMask, detectedMeta,
                    predictionOutputFile,
                    write_as_type=config.output_dtype,
                    th=0.7,
                    create_countors=False
                )
                print(f"  Saved prediction: {os.path.basename(predictionOutputFile)}")
                
                # Write confidence map to disk
                base_name = os.path.splitext(predictionOutputFile)[0]
                confidence_output_file = base_name + '_confidence.tif'
                confidence_meta = detectedMeta.copy()
                writeMaskToDisk(
                    confidenceMap, confidence_meta,
                    confidence_output_file,
                    write_as_type='float32',
                    th=0.0,  # No threshold for confidence map
                    create_countors=False
                )
                print(f"  Saved confidence map: {os.path.basename(confidence_output_file)}")
else:
    print("Processing loop example skipped. Set RUN_PROCESSING_LOOP_EXAMPLE = True and define file_pairs to run.")

#new on 12/02
# Functions for city-center area identification and analysis filtering

def identify_city_center_areas(pan_band, method='density_percentile', percentile=75, 
                                kernel_size=100, min_density_threshold=None):
    """
    Identify city-center areas based on urban density from PAN band.
    
    City-center areas are typically characterized by higher building density,
    which manifests as higher variance and intensity in the PAN band.
    
    Args:
        pan_band: PAN band as numpy array
        method: Method to identify city-center ('density_percentile', 'variance', 'intensity')
        percentile: Percentile threshold for density-based method (default: 75)
        kernel_size: Size of kernel for density calculation (default: 100 pixels)
        min_density_threshold: Optional absolute threshold for density (overrides percentile)
    
    Returns:
        city_center_mask: Boolean mask where True indicates city-center areas
        urban_density_map: Map of urban density values (0-1 scale)
    """
    from scipy import ndimage
    
    # Normalize PAN band to 0-1 range for consistent processing
    pan_normalized = (pan_band - pan_band.min()) / (pan_band.max() - pan_band.min() + 1e-10)
    
    if method == 'density_percentile':
        # Calculate local density using variance (urban areas have higher variance)
        # Use a moving window to calculate local variance
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = ndimage.convolve(pan_normalized, kernel, mode='constant')
        local_variance = ndimage.convolve((pan_normalized - local_mean)**2, kernel, mode='constant')
        
        # Combine intensity and variance as urban density indicator
        urban_density_map = 0.5 * pan_normalized + 0.5 * (local_variance / (local_variance.max() + 1e-10))
        
        # Threshold based on percentile
        if min_density_threshold is None:
            threshold = np.percentile(urban_density_map, percentile)
        else:
            threshold = min_density_threshold
        
        city_center_mask = urban_density_map >= threshold
        
    elif method == 'variance':
        # Use variance as indicator
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = ndimage.convolve(pan_normalized, kernel, mode='constant')
        local_variance = ndimage.convolve((pan_normalized - local_mean)**2, kernel, mode='constant')
        urban_density_map = local_variance / (local_variance.max() + 1e-10)
        
        if min_density_threshold is None:
            threshold = np.percentile(urban_density_map, percentile)
        else:
            threshold = min_density_threshold
        
        city_center_mask = urban_density_map >= threshold
        
    elif method == 'intensity':
        # Use intensity as indicator
        urban_density_map = pan_normalized
        
        if min_density_threshold is None:
            threshold = np.percentile(urban_density_map, percentile)
        else:
            threshold = min_density_threshold
        
        city_center_mask = urban_density_map >= threshold
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'density_percentile', 'variance', or 'intensity'")
    
    return city_center_mask, urban_density_map


def filter_analysis_to_city_center(mask, city_center_mask, urban_density_map=None):
    """
    Filter analysis results to only city-center areas.
    
    Args:
        mask: Prediction mask (2D array)
        city_center_mask: Boolean mask indicating city-center areas
        urban_density_map: Optional urban density map for stratification
    
    Returns:
        filtered_mask: Mask filtered to city-center areas only
        stats: Dictionary with statistics about filtered vs. full area
    """
    filtered_mask = mask.copy()
    filtered_mask[~city_center_mask] = 0
    
    # Calculate statistics
    total_pixels = mask.size
    city_center_pixels = np.sum(city_center_mask)
    city_center_fraction = city_center_pixels / total_pixels
    
    # Tree detection statistics
    full_area_trees = np.sum(mask > 0.5)
    city_center_trees = np.sum(filtered_mask > 0.5)
    
    stats = {
        'total_pixels': int(total_pixels),
        'city_center_pixels': int(city_center_pixels),
        'city_center_fraction': float(city_center_fraction),
        'full_area_tree_pixels': int(full_area_trees),
        'city_center_tree_pixels': int(city_center_trees),
        'full_area_tree_coverage': float(full_area_trees / total_pixels),
        'city_center_tree_coverage': float(city_center_trees / city_center_pixels) if city_center_pixels > 0 else 0.0,
    }
    
    return filtered_mask, stats


def calculate_stratified_metrics(mask, urban_density_map, n_strata=5, threshold=0.5):
    """
    Calculate prediction metrics stratified by urban density.
    
    Args:
        mask: Prediction mask (2D array)
        urban_density_map: Urban density map (0-1 scale)
        n_strata: Number of density strata to create (default: 5)
        threshold: Threshold for binary tree classification (default: 0.5)
    
    Returns:
        stratified_stats: Dictionary with metrics for each density stratum
    """
    binary_mask = (mask >= threshold).astype(bool)
    
    # Create density strata
    density_bins = np.linspace(0, 1, n_strata + 1)
    strata_labels = []
    strata_stats = []
    
    for i in range(n_strata):
        lower_bound = density_bins[i]
        upper_bound = density_bins[i + 1]
        
        # Create mask for this stratum
        if i == n_strata - 1:  # Include upper bound for last stratum
            stratum_mask = (urban_density_map >= lower_bound) & (urban_density_map <= upper_bound)
        else:
            stratum_mask = (urban_density_map >= lower_bound) & (urban_density_map < upper_bound)
        
        stratum_pixels = np.sum(stratum_mask)
        
        if stratum_pixels > 0:
            stratum_tree_pixels = np.sum(binary_mask & stratum_mask)
            stratum_tree_coverage = stratum_tree_pixels / stratum_pixels
            stratum_mean_density = np.mean(urban_density_map[stratum_mask])
            stratum_mean_confidence = np.mean(mask[stratum_mask]) if np.any(stratum_mask) else 0.0
        else:
            stratum_tree_pixels = 0
            stratum_tree_coverage = 0.0
            stratum_mean_density = (lower_bound + upper_bound) / 2
            stratum_mean_confidence = 0.0
        
        strata_labels.append(f"Density {i+1} ({lower_bound:.2f}-{upper_bound:.2f})")
        strata_stats.append({
            'density_range': (lower_bound, upper_bound),
            'pixels': int(stratum_pixels),
            'tree_pixels': int(stratum_tree_pixels),
            'tree_coverage': float(stratum_tree_coverage),
            'mean_urban_density': float(stratum_mean_density),
            'mean_prediction_confidence': float(stratum_mean_confidence),
        })
    
    return {
        'n_strata': n_strata,
        'strata': strata_stats,
        'strata_labels': strata_labels
    }


def print_model_coverage_transparency():
    """
    Print transparency statement about model coverage and limitations.
    """
    print("=" * 80)
    print("MODEL COVERAGE AND TRANSPARENCY STATEMENT")
    print("=" * 80)
    print()
    print("Model Training and Coverage:")
    print("  • This UNet model was trained on metro-wide areas covering diverse")
    print("    urban landscapes including city centers, suburbs, and peri-urban zones.")
    print("  • The model can generate predictions across the full metropolitan area.")
    print()
    print("Analysis Scope and Reliability:")
    print("  • Formal analysis and reporting are limited to city-center areas where")
    print("    predictions are most reliable.")
    print("  • City-center areas are identified based on urban density metrics")
    print("    derived from the PAN band (building density indicators).")
    print("  • Predictions in suburban and peri-urban areas may have lower reliability")
    print("    due to differences in training data distribution and landscape characteristics.")
    print()
    print("Confidence and Stratification:")
    print("  • Prediction confidence maps are available to identify areas with")
    print("    higher vs. lower prediction certainty.")
    print("  • Metrics are stratified by urban density to help readers understand")
    print("    where predictions are strong vs. weak.")
    print("  • Users should interpret results with awareness of these spatial variations.")
    print()
    print("Recommendations:")
    print("  • For policy and planning decisions, prioritize city-center results.")
    print("  • Use confidence maps to identify areas requiring validation.")
    print("  • Consider stratified metrics when comparing across different urban zones.")
    print("=" * 80)
    print()

# Example: Using city-center filtering and transparency statements
# This cell demonstrates how to incorporate city-center analysis into your workflow

# Print transparency statement
print_model_coverage_transparency()

# Example usage - only run if required variables are available
# Set RUN_EXAMPLE = True to run the example, or False to skip it
RUN_EXAMPLE = False  # Set to True when you have pan_band and detectedMask ready

if RUN_EXAMPLE:
    # Try to load pan_band if not already available
    if 'pan_band' not in locals():
        pan_file_path = None
        
        # Option 1: Try to use file_pairs if available
        if 'file_pairs' in locals() and 'cutouts_dir' in locals() and len(file_pairs) > 0:
            pan_file_path = os.path.join(cutouts_dir, file_pairs[0][0])
            print(f"Using PAN file from file_pairs: {pan_file_path}")
        
        # Option 2: Try to find PAN files in cutouts_dir
        elif 'cutouts_dir' in locals() and os.path.exists(cutouts_dir):
            pan_files = [f for f in os.listdir(cutouts_dir) 
                        if f.startswith('pan_') and f.endswith('.tif')]
            if pan_files:
                pan_file_path = os.path.join(cutouts_dir, pan_files[0])
                print(f"Using first PAN file found: {pan_file_path}")
        
        # Option 3: Try default cutouts directory
        elif os.path.exists("cutouts"):
            pan_files = [f for f in os.listdir("cutouts") 
                        if f.startswith('pan_') and f.endswith('.tif')]
            if pan_files:
                pan_file_path = os.path.join("cutouts", pan_files[0])
                print(f"Using first PAN file from default cutouts directory: {pan_file_path}")
        
        if pan_file_path and os.path.exists(pan_file_path):
            with rasterio.open(pan_file_path) as pan_src:
                pan_band = pan_src.read(1)
        else:
            print("Warning: Could not find PAN file. Please set pan_file_path manually or ensure cutouts_dir contains PAN files.")
            print("Example: pan_file_path = 'cutouts/pan_your_file.tif'")
            RUN_EXAMPLE = False
    
    # Check that detectedMask exists
    if 'detectedMask' not in locals():
        print("Warning: detectedMask not found. Please run detect_tree() first to generate the prediction mask.")
        RUN_EXAMPLE = False
    
    # Run the example if all requirements are met
    if RUN_EXAMPLE:

        # Identify city-center areas
        city_center_mask, urban_density_map = identify_city_center_areas(
            pan_band, 
            method='density_percentile', 
            percentile=75,  # Top 25% most urban areas
            kernel_size=100
        )
        
        # Filter predictions to city-center only
        filtered_mask, stats = filter_analysis_to_city_center(
            detectedMask, 
            city_center_mask, 
            urban_density_map
        )
        
        # Optional: Calculate stratified metrics by urban density
        stratified_metrics = calculate_stratified_metrics(
            detectedMask, 
            urban_density_map, 
            n_strata=5
        )
        
        # Print statistics
        print("\nCity-Center Filtering Statistics:")
        print(f"  City-center area: {stats['city_center_fraction']*100:.1f}% of total")
        print(f"  Tree coverage (full area): {stats['full_area_tree_coverage']*100:.2f}%")
        print(f"  Tree coverage (city-center): {stats['city_center_tree_coverage']*100:.2f}%")
        print("\nStratified Metrics by Urban Density:")
        for i, (label, stat) in enumerate(zip(stratified_metrics['strata_labels'], 
                                               stratified_metrics['strata'])):
            print(f"\n{label}:")
            print(f"  Pixels: {stat['pixels']:,}")
            print(f"  Tree coverage: {stat['tree_coverage']*100:.2f}%")
            print(f"  Mean prediction confidence: {stat['mean_prediction_confidence']:.3f}")
else:
    print("Example code skipped. Set RUN_EXAMPLE = True and ensure pan_band and detectedMask are available to run the example.")
    
# Visualization functions for confidence maps and city-center analysis

def visualize_city_center_analysis(pan_band, mask, city_center_mask, urban_density_map, 
                                   confidence_map=None, figsize=(15, 10)):
    """
    Create a comprehensive visualization showing city-center analysis.
    
    Args:
        pan_band: PAN band image
        mask: Prediction mask
        city_center_mask: Boolean mask for city-center areas
        urban_density_map: Urban density map
        confidence_map: Optional confidence map
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 3 if confidence_map is not None else 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. PAN band
    ax = axes[0]
    im = ax.imshow(pan_band, cmap='gray')
    ax.set_title('PAN Band (Input)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # 2. Urban density map
    ax = axes[1]
    im = ax.imshow(urban_density_map, cmap='hot')
    ax.set_title('Urban Density Map')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Urban Density')
    
    # 3. City-center mask overlay
    ax = axes[2]
    ax.imshow(pan_band, cmap='gray', alpha=0.7)
    overlay = city_center_mask.astype(float)
    overlay[~city_center_mask] = np.nan
    im = ax.imshow(overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('City-Center Areas (Red Overlay)')
    ax.axis('off')
    
    # 4. Full prediction mask
    ax = axes[3]
    im = ax.imshow(mask, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Full Area Predictions')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Prediction Confidence')
    
    # 5. City-center filtered predictions
    filtered_mask = mask.copy()
    filtered_mask[~city_center_mask] = 0
    ax = axes[4]
    im = ax.imshow(filtered_mask, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('City-Center Filtered Predictions')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Prediction Confidence')
    
    # 6. Confidence map (if provided)
    if confidence_map is not None:
        ax = axes[5]
        im = ax.imshow(confidence_map, cmap='plasma')
        ax.set_title('Prediction Confidence Map')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Confidence')
    
    plt.tight_layout()
    return fig


def create_city_center_report(mask, city_center_mask, urban_density_map, 
                               confidence_map=None, threshold=0.5):
    """
    Create a comprehensive report on city-center vs. full area analysis.
    
    Args:
        mask: Prediction mask
        city_center_mask: Boolean mask for city-center areas
        urban_density_map: Urban density map
        confidence_map: Optional confidence map
        threshold: Threshold for binary classification
    
    Returns:
        report: Dictionary with comprehensive statistics
    """
    # Filter to city-center
    filtered_mask, filter_stats = filter_analysis_to_city_center(
        mask, city_center_mask, urban_density_map
    )
    
    # Calculate stratified metrics
    stratified_metrics = calculate_stratified_metrics(
        mask, urban_density_map, n_strata=5, threshold=threshold
    )
    
    # Calculate confidence statistics if available
    confidence_stats = {}
    if confidence_map is not None:
        confidence_stats = {
            'mean_confidence_full': float(np.mean(confidence_map)),
            'mean_confidence_city_center': float(np.mean(confidence_map[city_center_mask])),
            'std_confidence_full': float(np.std(confidence_map)),
            'std_confidence_city_center': float(np.std(confidence_map[city_center_mask])),
            'min_confidence': float(np.min(confidence_map)),
            'max_confidence': float(np.max(confidence_map)),
        }
    
    report = {
        'filtering_stats': filter_stats,
        'stratified_metrics': stratified_metrics,
        'confidence_stats': confidence_stats if confidence_map is not None else None,
    }
    
    # Print report
    print("=" * 80)
    print("CITY-CENTER ANALYSIS REPORT")
    print("=" * 80)
    print("\n1. AREA COVERAGE:")
    print(f"   Total pixels: {filter_stats['total_pixels']:,}")
    print(f"   City-center pixels: {filter_stats['city_center_pixels']:,} "
          f"({filter_stats['city_center_fraction']*100:.1f}% of total)")
    
    print("\n2. TREE DETECTION:")
    print(f"   Full area tree pixels: {filter_stats['full_area_tree_pixels']:,}")
    print(f"   Full area tree coverage: {filter_stats['full_area_tree_coverage']*100:.2f}%")
    print(f"   City-center tree pixels: {filter_stats['city_center_tree_pixels']:,}")
    print(f"   City-center tree coverage: {filter_stats['city_center_tree_coverage']*100:.2f}%")
    
    if confidence_map is not None:
        print("\n3. PREDICTION CONFIDENCE:")
        print(f"   Mean confidence (full area): {confidence_stats['mean_confidence_full']:.3f}")
        print(f"   Mean confidence (city-center): {confidence_stats['mean_confidence_city_center']:.3f}")
        print(f"   Std confidence (full area): {confidence_stats['std_confidence_full']:.3f}")
        print(f"   Std confidence (city-center): {confidence_stats['std_confidence_city_center']:.3f}")
    
    print("\n4. STRATIFIED METRICS BY URBAN DENSITY:")
    for label, stat in zip(stratified_metrics['strata_labels'], 
                          stratified_metrics['strata']):
        print(f"\n   {label}:")
        print(f"     Pixels: {stat['pixels']:,}")
        print(f"     Tree coverage: {stat['tree_coverage']*100:.2f}%")
        print(f"     Mean prediction confidence: {stat['mean_prediction_confidence']:.3f}")
    
    print("\n" + "=" * 80)
    
    return report
# Integration example: Modified processing loop with city-center filtering
# This shows how to modify your main processing loop to include city-center analysis


# Example: Modified processing loop with city-center filtering and confidence maps
# Set these flags to control behavior
ENABLE_CITY_CENTER_FILTERING = True  # Set to True to filter to city-center areas
ENABLE_CONFIDENCE_MAPS = True  # Set to True to generate confidence maps
ENABLE_STRATIFIED_METRICS = True  # Set to True to calculate stratified metrics
CITY_CENTER_PERCENTILE = 75  # Percentile threshold for city-center identification
RUN_PROCESSING_LOOP_EXAMPLE = False  # Set to True to run the processing loop example

# Set cutouts directory (default to "cutouts" if not already defined)
if 'cutouts_dir' not in locals():
    cutouts_dir = CUTOUTS_DIR

# Print transparency statement at start
if ENABLE_CITY_CENTER_FILTERING:
    print_model_coverage_transparency()

# Modified processing loop (example structure):
# Only run if RUN_PROCESSING_LOOP_EXAMPLE is True and file_pairs is available
if RUN_PROCESSING_LOOP_EXAMPLE:
    # Auto-create file_pairs from cutouts directory if not already defined
    if 'file_pairs' not in locals() or not file_pairs:
        print(f"Creating file_pairs from {cutouts_dir} directory...")
        if not os.path.exists(cutouts_dir):
            print(f"Error: Directory '{cutouts_dir}' does not exist!")
        else:
            file_pairs = []
            # Get all PAN files
            all_files = os.listdir(cutouts_dir)
            pan_files = [f for f in all_files if f.startswith('pan_') and f.endswith('.tif') and 'atlanta' not in f.lower()]
            
            for pan_file in pan_files:
                # Find corresponding NDVI file (replace 'pan_' with 'ndvi_')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
                pan_path = os.path.join(cutouts_dir, pan_file)
                ndvi_path = os.path.join(cutouts_dir, ndvi_file)
                
                if os.path.exists(ndvi_path):
                    file_pairs.append((pan_path, ndvi_path))
            
            print(f"Found {len(file_pairs)} file pairs in {cutouts_dir}")
            if len(file_pairs) == 0:
                print("Warning: No matching PAN/NDVI file pairs found!")
                print(f"PAN files found: {len(pan_files)}")
    
    if 'file_pairs' in locals() and len(file_pairs) > 0:
        for pan_file, ndvi_file in file_pairs:
            # Extract filename from full path for output file naming
            pan_filename = os.path.basename(pan_file)
            predictionOutputFile = os.path.join(cutouts_dir, f"pred_{pan_filename}")
            
            # Skip if already processed (unless overwrite is enabled)
            if os.path.exists(predictionOutputFile) and not getattr(config, 'overwrite_analysed_files', False):
                print(f"Skipping {pan_filename} - already processed")
                continue
            
            print(f"\nProcessing: {pan_filename}")
            
            with rasterio.open(pan_file) as pan_img, \
                 rasterio.open(ndvi_file) as ndvi_img:
                
                pan_band = pan_img.read(1)
                ndvi_band = ndvi_img.read(1)
                
                # Detect trees (with optional confidence map)
                if ENABLE_CONFIDENCE_MAPS:
                    detectedMask, detectedMeta, confidenceMap = detect_tree(
                        ndvi_band, pan_band,
                        width=config.WIDTH,
                        height=config.HEIGHT,
                        stride=config.STRIDE,
                        return_confidence=True
                    )
                else:
                    detectedMask, detectedMeta = detect_tree(
                        ndvi_band, pan_band,
                        width=config.WIDTH,
                        height=config.HEIGHT,
                        stride=config.STRIDE,
                        return_confidence=False
                    )
                    confidenceMap = None
                
                # Identify city-center areas
                if ENABLE_CITY_CENTER_FILTERING:
                    city_center_mask, urban_density_map = identify_city_center_areas(
                        pan_band,
                        method='density_percentile',
                        percentile=CITY_CENTER_PERCENTILE,
                        kernel_size=100
                    )
                    
                    # Filter predictions to city-center
                    filtered_mask, filter_stats = filter_analysis_to_city_center(
                        detectedMask,
                        city_center_mask,
                        urban_density_map
                    )
                    
                    # Use filtered mask for analysis/reporting
                    analysis_mask = filtered_mask
                    
                    # Generate report
                    report = create_city_center_report(
                        detectedMask,
                        city_center_mask,
                        urban_density_map,
                        confidenceMap
                    )
                    
                    # Optional: Visualize
                    # fig = visualize_city_center_analysis(
                    #     pan_band, detectedMask, city_center_mask, 
                    #     urban_density_map, confidenceMap
                    # )
                    # plt.savefig(f'{output_dir}/city_center_analysis_{pan_file}.png')
                    # plt.close()
                else:
                    # Use full mask for analysis
                    analysis_mask = detectedMask
                
                # Write full predictions to disk (always save full predictions)
                writeMaskToDisk(
                    detectedMask, detectedMeta,
                    predictionOutputFile,
                    write_as_type=config.output_dtype,
                    th=0.7,
                    create_countors=False
                )
                
                # Optionally write city-center filtered predictions
                if ENABLE_CITY_CENTER_FILTERING:
                    city_center_output_file = predictionOutputFile.replace(
                        config.output_prefix, 
                        config.output_prefix + '_citycenter'
                    )
                    writeMaskToDisk(
                        filtered_mask, detectedMeta,
                        city_center_output_file,
                        write_as_type=config.output_dtype,
                        th=0.7,
                        create_countors=False
                    )
                
                # Optionally write confidence map
                if ENABLE_CONFIDENCE_MAPS and confidenceMap is not None:
                    confidence_output_file = predictionOutputFile.replace(
                        config.output_prefix,
                        config.output_prefix + '_confidence'
                    )
                    confidence_meta = detectedMeta.copy()
                    writeMaskToDisk(
                        confidenceMap, confidence_meta,
                        confidence_output_file,
                        write_as_type='float32',
                        th=0.0,  # No threshold for confidence map
                        create_countors=False
                    )
else:
    print("Processing loop example skipped. Set RUN_PROCESSING_LOOP_EXAMPLE = True and define file_pairs to run.")

# Display extracted image
#sampleImage = 'pan_0-0.tif'
#fn = os.path.join(config.output_dir, config.output_prefix + sampleImage )
#predicted_img = rasterio.open(fn)
#p = predicted_img.read()
#np.unique(p, return_counts=True)
#plt.imshow(p[0])

# List all files in the output directory

# Display extracted image and its prediction
#sampleImage = 'pan_0-15000.tif'
#fn = os.path.join(config.output_dir, config.output_prefix + sampleImage)

#with rasterio.open(fn) as img:
#    data = img.read()
    
    # Create a figure with two subplots
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original data
#    im1 = ax1.imshow(data[0], cmap='viridis')
#    ax1.set_title('Original Data')
#    plt.colorbar(im1, ax=ax1, label='Values')
    
    # Plot thresholded version to see potential predictions
#    thresholded = data[0].copy()
#    threshold = 0.5
#    thresholded[thresholded <
#    threshold] = 0
#    thresholded[thresholded >= threshold] = 1
    
#    im2 = ax2.imshow(thresholded, cmap='binary')
#    ax2.set_title(f'Thresholded (>{threshold})')
#    plt.colorbar(im2, ax=ax2, label='Binary Prediction')
    
    # Print some statistics
#    print(f"Value range: {data.min():.3f} to {data.max():.3f}")
#    print(f"Image shape: {data.shape}")
#    print(f"Unique values: {np.unique(data).shape[0]} values")

# Find all input files in the cutouts directory

# Find all input files in the cutouts directory

import time
# Find corresponding pairs of PAN and NDVI files
cutouts_dir = CUTOUTS_DIR

# Verify cutouts directory exists
print("="*80)
print("VERIFYING PATHS AND DIRECTORIES")
print("="*80)
print(f"BASE_DIR: {BASE_DIR}")
print(f"CUTOUTS_DIR: {cutouts_dir}")
if not os.path.exists(cutouts_dir):
    raise FileNotFoundError(f"ERROR: Cutouts directory does not exist: {cutouts_dir}\n"
                           f"Please verify the path is correct for Sherlock.")
print(f"✓ Cutouts directory exists: {cutouts_dir}")
print("="*80)
print()

file_pairs = []

# Get unique tile pairs by looking for PAN and NDVI matches
unique_pairs = {}
all_files = os.listdir(cutouts_dir)

for f in all_files:
    if f.endswith('.tif') and not f.startswith('pred_') and 'atlanta' not in f.lower():
        if f.startswith('pan_'):
            base = f.replace('pan_', '')
            if base not in unique_pairs: unique_pairs[base] = {}
            unique_pairs[base]['pan'] = f
        elif f.startswith('ndvi_'):
            base = f.replace('ndvi_', '')
            if base not in unique_pairs: unique_pairs[base] = {}
            unique_pairs[base]['ndvi'] = f

# Only include pairs that have both PAN and NDVI
file_pairs = sorted([(v['pan'], v['ndvi']) for k, v in unique_pairs.items() if 'pan' in v and 'ndvi' in v])

# --- Status Summary ---
total = len(file_pairs)
completed_list = [pan for pan, _ in file_pairs if os.path.exists(os.path.join(cutouts_dir, f"pred_{pan}"))]
completed = len(completed_list)
remaining = total - completed

print("="*60)
print("PREDICTION STATUS SUMMARY")
print("="*60)
print(f"Total Unique Tiles: {total}")
print(f"Already Completed:  {completed} ({completed/total*100:.1f}%)")
print(f"Remaining to Do:    {remaining}")
print("="*60)
print()

# Create progress bar for overall processing
overall_progress = tqdm(file_pairs, desc="Overall Progress", position=0)
processed_files = []
failed_files = []
start_time = time.time()

for pan_file, ndvi_file in overall_progress:
    file_start_time = time.time()
    predictionOutputFile = os.path.join(cutouts_dir, f"pred_{pan_file}")
    
    try:
        if not os.path.isfile(predictionOutputFile) or config.overwrite_analysed_files: 
            with rasterio.open(os.path.join(cutouts_dir, pan_file)) as pan_img, \
                 rasterio.open(os.path.join(cutouts_dir, ndvi_file)) as ndvi_img:
                
                overall_progress.set_postfix({'Current File': pan_file})
                print(f"\nProcessing pair: {pan_file} and {ndvi_file}")
                
                # Read the bands and verify they're not empty
                pan_band = pan_img.read(1)   
                ndvi_band = ndvi_img.read(1)
                
                # Add input validation checks
                if pan_band.size == 0 or ndvi_band.size == 0:
                    print(f"Warning: Empty input bands for {pan_file}")
                    failed_files.append((pan_file, "Empty input bands"))
                    continue
                    
                print(f"Input stats:")
                print(f"PAN - Shape: {pan_band.shape}, Range: [{pan_band.min():.3f}, {pan_band.max():.3f}]")
                print(f"NDVI - Shape: {ndvi_band.shape}, Range: [{ndvi_band.min():.3f}, {ndvi_band.max():.3f}]")

                # Check for NaN or infinite values
                if np.any(np.isnan(pan_band)) or np.any(np.isnan(ndvi_band)):
                    print("Warning: Input contains NaN values")
                    # Optional: Clean NaN values
                    pan_band = np.nan_to_num(pan_band)
                    ndvi_band = np.nan_to_num(ndvi_band)

                # Get metadata from PAN image
                meta = pan_img.meta.copy()
                
                detectedMask, detectedMeta = detect_tree(ndvi_band, pan_band, 
                                                        width=config.WIDTH, 
                                                        height=config.HEIGHT, 
                                                        stride=config.STRIDE, 
                                                        normalize=True)
                
                # Validate prediction output
                if detectedMask.size == 0 or np.all(detectedMask == 0):
                    print("Warning: Empty prediction mask generated")
                    failed_files.append((pan_file, "Empty prediction"))
                    continue
                
                print(f"Prediction stats:")
                print(f"Shape: {detectedMask.shape}")
                print(f"Range: [{detectedMask.min():.3f}, {detectedMask.max():.3f}]")
                print(f"Mean: {detectedMask.mean():.3f}")
                print(f"Unique values: {np.unique(detectedMask).size}")
                
                # Update metadata with the source image's transform and CRS
                detectedMeta.update({
                    'transform': meta['transform'],
                    'crs': meta['crs'],
                    'driver': 'GTiff',  # Explicitly set the driver
                    'dtype': 'float32'  # Ensure correct dtype
                })
                
                # Write the mask to the prediction file
                writeMaskToDisk(detectedMask, detectedMeta, predictionOutputFile, 
                               write_as_type=config.output_dtype, 
                               th=0.7, 
                               create_countors=False)
                
                # Verify the written file
                if os.path.exists(predictionOutputFile):
                    with rasterio.open(predictionOutputFile) as written_file:
                        written_data = written_file.read(1)
                        if np.all(written_data == 0):
                            print("Warning: Written file contains all zeros")
                        else:
                            print(f"Successfully wrote non-empty prediction to {predictionOutputFile}")
                
                file_time = time.time() - file_start_time
                processed_files.append((pan_file, file_time))
                
                # Calculate and display time estimates
                avg_time_per_file = sum(t for _, t in processed_files) / len(processed_files)
                files_remaining = len(file_pairs) - len(processed_files) - len(failed_files)
                est_time_remaining = avg_time_per_file * files_remaining
                
                print(f"\nFile completed in {file_time:.1f} seconds")
                print(f"Estimated time remaining: {est_time_remaining/60:.1f} minutes")
                print(f"Files processed: {len(processed_files)}/{len(file_pairs)}")
                
                # Clear memory
                import gc
                gc.collect()
                tf.keras.backend.clear_session()
                
        else:
            print(f'\nAlready processed: {pan_file}')
            processed_files.append((pan_file, 0))  # Add to processed with 0 time
            
    except Exception as e:
        print(f"\nError processing {pan_file}: {str(e)}")
        failed_files.append((pan_file, str(e)))
        continue

# Final summary
total_time = time.time() - start_time
print("\n=== Processing Complete ===")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Successfully processed: {len(processed_files)} files")
print(f"Failed: {len(failed_files)} files")

if failed_files:
    print("\nFailed files:")
    for file, error in failed_files:
        print(f"- {file}: {error}")

# Calculate statistics
if processed_files:
    processing_times = [t for _, t in processed_files if t > 0]
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\nAverage processing time per file: {avg_time:.1f} seconds")

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import LinearSegmentedColormap

def analyze_and_display_comparison(ndvi_file, pred_file, figsize=(20, 5)):
    """
    Display NDVI and prediction images side by side
    """
    try:
        with rasterio.open(ndvi_file) as ndvi_src, \
             rasterio.open(pred_file) as pred_src:
            
            ndvi_img = ndvi_src.read(1)
            pred_img = pred_src.read(1)
            
            # Clip extreme NDVI values for visualization
            ndvi_viz = np.clip(ndvi_img, -1, 1)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Display NDVI image with custom colormap
            colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
                     '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837']
            ndvi_cmap = LinearSegmentedColormap.from_list('ndvi_cmap', colors)
            
            im1 = ax1.imshow(ndvi_viz, cmap=ndvi_cmap, vmin=-1, vmax=1)
            ax1.set_title('NDVI Image (Clipped to [-1,1])')
            plt.colorbar(im1, ax=ax1, label='NDVI Value')
            
            # Display prediction
            im2 = ax2.imshow(pred_img, cmap='viridis')
            ax2.set_title('Prediction')
            plt.colorbar(im2, ax=ax2, label='Prediction Confidence')
            
            plt.tight_layout()
            # Save figure instead of showing (for batch processing)
            output_path = ndvi_file.replace('.tif', '_comparison.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Print statistics about vegetation
            veg_threshold = 0.2
            veg_pixels = np.sum(ndvi_img > veg_threshold)
            total_pixels = ndvi_img.size
            print(f"\nVegetation Analysis:")
            print(f"Pixels with NDVI > {veg_threshold}: {veg_pixels:,}")
            print(f"Percentage of vegetation coverage: {(veg_pixels/total_pixels)*100:.1f}%")
            print(f"\nNDVI Statistics:")
            print(f"Values in normal range [-1,1]: {np.sum((ndvi_img >= -1) & (ndvi_img <= 1))/total_pixels*100:.1f}%")
            print(f"Number of outliers: {np.sum((ndvi_img < -1) | (ndvi_img > 1)):,}")
            
    except Exception as e:
        print(f"Error processing images: {e}")

def find_image_sets(cutouts_dir):
    """
    Find matching sets of NDVI and prediction images
    """
    if not os.path.exists(cutouts_dir):
        print(f"Directory {cutouts_dir} does not exist!")
        return []
    
    all_files = os.listdir(cutouts_dir)
    tif_files = [f for f in all_files if f.endswith('.tif')]
    
    #print(f"Found {len(tif_files)} TIF files in {cutouts_dir}")
    
    # Group files by type
    ndvi_files = [f for f in tif_files if 'ndvi' in f.lower() and not f.startswith('pred_')]
    pred_files = [f for f in tif_files if f.startswith('pred_')]
    
    #print(f"NDVI files: {len(ndvi_files)}")
    #print(f"Prediction files: {len(pred_files)}")
    
    # Try to match files
    image_sets = []
    
    for pred_file in pred_files:
        # Extract base name from prediction file
        base_name = pred_file.replace('pred_', '')
        
        # Find corresponding NDVI file
        ndvi_match = None
        
        # Find NDVI file
        ndvi_base = base_name.replace('pan', 'ndvi')
        if ndvi_base in ndvi_files:
            ndvi_match = ndvi_base
        
        # If not found, try pattern matching
        if not ndvi_match:
            for ndvi_file in ndvi_files:
                if any(part in ndvi_file for part in base_name.split('_')):
                    ndvi_match = ndvi_file
                    break
        
        if ndvi_match:
            image_sets.append({
                'ndvi': ndvi_match,
                'pred': pred_file
            })
          #  print(f"Found set: NDVI={ndvi_match}, PRED={pred_file}")
    
    return image_sets

def analyze_multiple_sets(cutouts_dir, n_display=4):
    """
    Display multiple sets of NDVI and predictions
    """
    image_sets = find_image_sets(cutouts_dir)
    
    if not image_sets:
        print("No matching image sets found!")
        return
    
    print(f"\nFound {len(image_sets)} complete image sets")
    
    for i, img_set in enumerate(image_sets[:n_display]):
        print(f"\n{'='*60}")
        print(f"Analyzing image set {i+1}/{min(n_display, len(image_sets))}")
        print(f"{'='*60}")
        
        ndvi_path = os.path.join(cutouts_dir, img_set['ndvi'])
        pred_path = os.path.join(cutouts_dir, img_set['pred'])
        
        analyze_and_display_comparison(ndvi_path, pred_path)

if __name__ == "__main__":
    # Set the cutouts directory
    cutouts_dir = CUTOUTS_DIR
    
    # Run the analysis (commented out for batch processing)
    # analyze_multiple_sets(cutouts_dir, n_display=4)

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import LinearSegmentedColormap

def analyze_and_display_comparison(ndvi_file, pred_file, figsize=(20, 5)):
    """
    Display NDVI and prediction images side by side
    """
    try:
        with rasterio.open(ndvi_file) as ndvi_src, \
             rasterio.open(pred_file) as pred_src:
            
            ndvi_img = ndvi_src.read(1)
            pred_img = pred_src.read(1)
            
            # Clip extreme NDVI values for visualization
            ndvi_viz = np.clip(ndvi_img, -1, 1)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Display NDVI image with custom colormap
            colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
                     '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837']
            ndvi_cmap = LinearSegmentedColormap.from_list('ndvi_cmap', colors)
            
            im1 = ax1.imshow(ndvi_viz, cmap=ndvi_cmap, vmin=-1, vmax=1)
            ax1.set_title('NDVI Image (Clipped to [-1,1])')
            plt.colorbar(im1, ax=ax1, label='NDVI Value')
            
            # Display prediction
            im2 = ax2.imshow(pred_img, cmap='viridis')
            ax2.set_title('Prediction')
            plt.colorbar(im2, ax=ax2, label='Prediction Confidence')
            
            plt.tight_layout()
            # Save figure instead of showing (for batch processing)
            output_path = ndvi_file.replace('.tif', '_comparison.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Print statistics about vegetation
            veg_threshold = 0.2
            veg_pixels = np.sum(ndvi_img > veg_threshold)
            total_pixels = ndvi_img.size
            print(f"\nVegetation Analysis:")
            print(f"Pixels with NDVI > {veg_threshold}: {veg_pixels:,}")
            print(f"Percentage of vegetation coverage: {(veg_pixels/total_pixels)*100:.1f}%")
            print(f"\nNDVI Statistics:")
            print(f"Values in normal range [-1,1]: {np.sum((ndvi_img >= -1) & (ndvi_img <= 1))/total_pixels*100:.1f}%")
            print(f"Number of outliers: {np.sum((ndvi_img < -1) | (ndvi_img > 1)):,}")
            
    except Exception as e:
        print(f"Error processing images: {e}")

def find_image_sets(cutouts_dir):
    """
    Find matching sets of NDVI and prediction images
    """
    if not os.path.exists(cutouts_dir):
        print(f"Directory {cutouts_dir} does not exist!")
        return []
    
    all_files = os.listdir(cutouts_dir)
    tif_files = [f for f in all_files if f.endswith('.tif')]
    
    #print(f"Found {len(tif_files)} TIF files in {cutouts_dir}")
    
    # Group files by type
    ndvi_files = [f for f in tif_files if 'ndvi' in f.lower() and not f.startswith('pred_')]
    pred_files = [f for f in tif_files if f.startswith('pred_')]
    
    #print(f"NDVI files: {len(ndvi_files)}")
    #print(f"Prediction files: {len(pred_files)}")
    
    # Try to match files
    image_sets = []
    
    for pred_file in pred_files:
        # Extract base name from prediction file
        base_name = pred_file.replace('pred_', '')
        
        # Find corresponding NDVI file
        ndvi_match = None
        
        # Find NDVI file
        ndvi_base = base_name.replace('pan', 'ndvi')
        if ndvi_base in ndvi_files:
            ndvi_match = ndvi_base
        
        # If not found, try pattern matching
        if not ndvi_match:
            for ndvi_file in ndvi_files:
                if any(part in ndvi_file for part in base_name.split('_')):
                    ndvi_match = ndvi_file
                    break
        
        if ndvi_match:
            image_sets.append({
                'ndvi': ndvi_match,
                'pred': pred_file
            })
            #print(f"Found set: NDVI={ndvi_match}, PRED={pred_file}")
    
    return image_sets

def analyze_multiple_sets(cutouts_dir, n_display=4):
    """
    Display multiple sets of NDVI and predictions
    """
    image_sets = find_image_sets(cutouts_dir)
    
    if not image_sets:
        print("No matching image sets found!")
        return
    
    print(f"\nFound {len(image_sets)} complete image sets")
    
    for i, img_set in enumerate(image_sets[:n_display]):
        print(f"\n{'='*60}")
        print(f"Analyzing image set {i+1}/{min(n_display, len(image_sets))}")
        print(f"{'='*60}")
        
        ndvi_path = os.path.join(cutouts_dir, img_set['ndvi'])
        pred_path = os.path.join(cutouts_dir, img_set['pred'])
        
        analyze_and_display_comparison(ndvi_path, pred_path)

if __name__ == "__main__":
    # Set the cutouts directory
    cutouts_dir = CUTOUTS_DIR
    
    # Run the analysis (commented out for batch processing)
    # analyze_multiple_sets(cutouts_dir, n_display=10)

def get_city_from_filename(filename):
    """
    Extract city name from prediction filename
    Examples:
    - pred_ndvi_30000-10000_atlanta_064_216.tif -> Atlanta
    - pred_pan_5000-0.tif -> Unknown (no city in name)
    """
    # List of known cities (lowercase)
    known_cities = ['atlanta', 'austin', 'bloomington', 'cupertino', 'surrey']
    
    filename_lower = filename.lower()
    
    for city in known_cities:
        if city in filename_lower:
            return city.title()
    
    return "Unknown"
# Check which files are being labeled as "Unknown"
cutouts_dir = CUTOUTS_DIR
pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])

print("Files classified as 'Unknown':")
print("="*60)
for f in pred_files:
    city = get_city_from_filename(f)
    if city == "Unknown":
        print(f)
        
print("\n" + "="*60)
print(f"\nTotal files: {len(pred_files)}")
print(f"Unknown files: {sum(1 for f in pred_files if get_city_from_filename(f) == 'Unknown')}")

# Additional diagnostics - show coordinates of unknown files and compare with known city files
cutouts_dir = CUTOUTS_DIR
pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])

print("\n" + "="*80)
print("COORDINATE ANALYSIS")
print("="*80)

# Get coordinates from some known city files
known_cities_coords = {}
for city in ['atlanta', 'austin', 'bloomington', 'cupertino', 'surrey']:
    city_files = [f for f in pred_files if city in f.lower()]
    if city_files:
        # Get coordinates from first file of this city
        sample_file = city_files[0]
        file_path = os.path.join(cutouts_dir, sample_file)
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            known_cities_coords[city.title()] = (center_lon, center_lat)
            print(f"{city.title():15s}: Lon={center_lon:10.4f}, Lat={center_lat:9.4f} (from {sample_file})")

print("\n" + "-"*80)
print("Unknown files coordinates:")
print("-"*80)

unknown_files = [f for f in pred_files if get_city_from_filename(f) == 'Unknown']
unknown_coords = []

for f in unknown_files:
    file_path = os.path.join(cutouts_dir, f)
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
        unknown_coords.append((f, center_lon, center_lat))
        print(f"{f:35s}: Lon={center_lon:10.4f}, Lat={center_lat:9.4f}")
        
        # Find closest known city by distance
        if known_cities_coords:
            min_dist = float('inf')
            closest_city = None
            for city, (known_lon, known_lat) in known_cities_coords.items():
                # Simple Euclidean distance (good enough for comparison)
                dist = ((center_lon - known_lon)**2 + (center_lat - known_lat)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_city = city
            print(f"  → Closest to {closest_city} (distance: {min_dist:.4f})")

print("\n" + "="*80)
print("SUGGESTION: These unknown files are likely from:", closest_city if 'closest_city' in locals() else "UNKNOWN")
print("="*80)


# Investigate and identify "Unknown" files using geospatial data
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

def load_city_boundaries(area_dir='Preprocessing/input/area'):
    """Load all city boundary shapefiles"""
    city_boundaries = {}
    
    if not os.path.exists(area_dir):
        print(f"Warning: Area directory {area_dir} not found")
        return city_boundaries
    
    shp_files = [f for f in os.listdir(area_dir) if f.endswith('.shp')]
    
    for shp_file in shp_files:
        city_name = shp_file.split('_')[0].title()
        shp_path = os.path.join(area_dir, shp_file)
        
        try:
            gdf = gpd.read_file(shp_path)
            if city_name not in city_boundaries:
                city_boundaries[city_name] = []
            city_boundaries[city_name].append(gdf)
        except Exception as e:
            print(f"Error loading {shp_file}: {e}")
    
    # Merge multiple boundaries for the same city using pd.concat
    for city in list(city_boundaries.keys()):
        if len(city_boundaries[city]) > 1:
            city_boundaries[city] = pd.concat(city_boundaries[city], ignore_index=True)
        else:
            city_boundaries[city] = city_boundaries[city][0]
    
    return city_boundaries


def identify_city_by_location(raster_path, city_boundaries):
    """
    Identify city by checking if raster bounds intersect with city boundaries
    """
    with rasterio.open(raster_path) as src:
        # Get raster bounds and CRS
        bounds = src.bounds
        raster_crs = src.crs
        
        # Create a box geometry from the bounds
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        raster_gdf = gpd.GeoDataFrame([1], geometry=[raster_box], crs=raster_crs)
        
        # Check each city boundary
        for city_name, boundary_gdf in city_boundaries.items():
            try:
                # Reproject raster bounds to match boundary CRS
                raster_reproj = raster_gdf.to_crs(boundary_gdf.crs)
                
                # Check for intersection
                if raster_reproj.intersects(boundary_gdf.unary_union).any():
                    return city_name, (bounds.left, bounds.bottom, bounds.right, bounds.top), str(raster_crs)
            except Exception as e:
                print(f"Error checking {city_name}: {e}")
                continue
    
    return None, None, None


# Load city boundaries
print("Loading city boundaries...")
city_boundaries = load_city_boundaries('Preprocessing/input/area')
print(f"Loaded boundaries for: {', '.join(city_boundaries.keys())}\n")

# Check unknown files
cutouts_dir = CUTOUTS_DIR
pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])

unknown_files = [f for f in pred_files if get_city_from_filename(f) == 'Unknown']

print("="*80)
print(f"Found {len(unknown_files)} files without city names in filename")
print("="*80)

if unknown_files and city_boundaries:
    print("\nAttempting to identify cities using geospatial coordinates...\n")
    
    identifications = {}
    
    for f in unknown_files:
        file_path = os.path.join(cutouts_dir, f)
        city, bounds, crs = identify_city_by_location(file_path, city_boundaries)
        
        identifications[f] = city
        
        print(f"File: {f}")
        print(f"  → Identified as: {city if city else 'UNKNOWN (no match)'}")
        if bounds:
            print(f"  → Bounds: ({bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f})")
            print(f"  → CRS: {crs}")
        print()
    
    # Summary
    print("="*80)
    print("IDENTIFICATION SUMMARY")
    print("="*80)
    identified = sum(1 for v in identifications.values() if v is not None)
    print(f"Successfully identified: {identified}/{len(unknown_files)}")
    
    if identified > 0:
        print("\nIdentified cities:")
        city_counts = {}
        for city in identifications.values():
            if city:
                city_counts[city] = city_counts.get(city, 0) + 1
        for city, count in sorted(city_counts.items()):
            print(f"  {city}: {count} files")
            
else:
    print("\nShowing unknown filenames:")
    for f in unknown_files:
        print(f"  {f}")

def get_city_from_filename(filename):
    """
    Extract city name from prediction filename
    Examples:
    - pred_ndvi_30000-10000_atlanta_064_216.tif -> Atlanta
    - pred_pan_5000-0.tif -> Unknown (no city in name)
    """
    # List of known cities (lowercase)
    known_cities = ['atlanta', 'austin', 'bloomington', 'cupertino', 'surrey']
    
    filename_lower = filename.lower()
    
    for city in known_cities:
        if city in filename_lower:
            return city.title()
    
    return "Unknown"


def analyze_all_predictions(cutouts_dir, group_by_city=True):
    """
    Analyze all prediction images in the directory and calculate total trees
    If group_by_city=True, trees are counted per city (extracted from filenames)
    """
    pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])
    
    total_trees = 0
    total_area = 0
    results = []
    city_stats = {}
    
    print(f"\nAnalyzing {len(pred_files)} prediction files...")
    
    for pred_file in tqdm(pred_files):
        pred_path = os.path.join(cutouts_dir, pred_file)
        
        with rasterio.open(pred_path) as src:
            pred_img = src.read(1)
            
            # Determine city from filename
            city = "Unknown"
            if group_by_city:
                city = get_city_from_filename(pred_file)
            
            # Create binary mask for tree counting
            binary_pred = pred_img > 0.5
            labeled_array, num_trees = measure.label(binary_pred, return_num=True)
            
            # Calculate area coverage
            area_coverage = np.sum(binary_pred) / binary_pred.size * 100
            
            total_trees += num_trees
            total_area += area_coverage
            
            # Update city statistics
            if city not in city_stats:
                city_stats[city] = {
                    'trees': 0,
                    'area': 0,
                    'files': 0,
                    'file_list': []
                }
            
            city_stats[city]['trees'] += num_trees
            city_stats[city]['area'] += area_coverage
            city_stats[city]['files'] += 1
            city_stats[city]['file_list'].append(pred_file)
            
            results.append({
                'file': pred_file,
                'city': city,
                'trees': num_trees,
                'coverage': area_coverage
            })
    
    # Print overall summary statistics
    print("\n" + "="*60)
    print("=== OVERALL SUMMARY STATISTICS ===")
    print("="*60)
    print(f"Total number of trees detected: {total_trees:,}")
    print(f"Average trees per image: {total_trees/len(pred_files):,.1f}")
    print(f"Average tree coverage: {total_area/len(pred_files):.2f}%")
    
    # Print city-wise statistics
    if group_by_city and city_stats:
        print("\n" + "="*60)
        print("=== TREE COUNT BY CITY ===")
        print("="*60)
        
        # Sort cities by tree count
        sorted_cities = sorted(city_stats.items(), key=lambda x: x[1]['trees'], reverse=True)
        
        for city, stats in sorted_cities:
            print(f"\n{city}:")
            print(f"  Total trees: {stats['trees']:,}")
            print(f"  Number of image tiles: {stats['files']}")
            print(f"  Average trees per tile: {stats['trees']/stats['files']:,.1f}")
            print(f"  Average coverage per tile: {stats['area']/stats['files']:.2f}%")
    
    # Find images with most and least trees
    max_trees = max(results, key=lambda x: x['trees'])
    min_trees = min(results, key=lambda x: x['trees'])
    
    print("\n" + "="*60)
    print("=== EXTREMES ===")
    print("="*60)
    print(f"\nImage with most trees: {max_trees['file']}")
    print(f"City: {max_trees['city']}")
    print(f"Number of trees: {max_trees['trees']:,}")
    
    print(f"\nImage with least trees: {min_trees['file']}")
    print(f"City: {min_trees['city']}")
    print(f"Number of trees: {min_trees['trees']:,}")
    
    return results, city_stats

# Run the analysis with city grouping
import pandas as pd
from skimage import measure

cutouts_dir = CUTOUTS_DIR
results, city_stats = analyze_all_predictions(cutouts_dir, group_by_city=True)

# Re-run analysis with geospatial matching for unknown files
import pandas as pd
from skimage import measure

def get_city_with_geospatial_fallback(filename, filepath, known_cities_coords):
    """
    Get city from filename, with geospatial fallback for unknown files
    """
    # First try filename
    city = get_city_from_filename(filename)
    
    # If unknown, try geospatial matching
    if city == "Unknown" and known_cities_coords:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            
            # Find closest known city
            min_dist = float('inf')
            closest_city = None
            for city_name, (known_lon, known_lat) in known_cities_coords.items():
                dist = ((center_lon - known_lon)**2 + (center_lat - known_lat)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_city = city_name
            
            # If very close (within ~0.5 degrees, about 50km), assign that city
            if min_dist < 0.5:
                city = closest_city
    
    return city


# Build reference coordinates from known files
cutouts_dir = CUTOUTS_DIR
pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])

known_cities_coords = {}
for city in ['atlanta', 'austin', 'bloomington', 'cupertino', 'surrey']:
    city_files = [f for f in pred_files if city in f.lower()]
    if city_files:
        sample_file = city_files[0]
        file_path = os.path.join(cutouts_dir, sample_file)
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            known_cities_coords[city.title()] = (center_lon, center_lat)

print("Reference coordinates loaded for:", ", ".join(known_cities_coords.keys()))

# Run analysis with geospatial fallback
total_trees = 0
total_area = 0
results = []
city_stats = {}

print(f"\nAnalyzing {len(pred_files)} prediction files with geospatial fallback...")

for pred_file in tqdm(pred_files):
    pred_path = os.path.join(cutouts_dir, pred_file)
    
    with rasterio.open(pred_path) as src:
        pred_img = src.read(1)
        
        # Determine city with geospatial fallback
        city = get_city_with_geospatial_fallback(pred_file, pred_path, known_cities_coords)
        
        # Create binary mask for tree counting
        binary_pred = pred_img > 0.5
        labeled_array, num_trees = measure.label(binary_pred, return_num=True)
        
        # Calculate area coverage
        area_coverage = np.sum(binary_pred) / binary_pred.size * 100
        
        total_trees += num_trees
        total_area += area_coverage
        
        # Update city statistics
        if city not in city_stats:
            city_stats[city] = {
                'trees': 0,
                'area': 0,
                'files': 0,
                'file_list': []
            }
        
        city_stats[city]['trees'] += num_trees
        city_stats[city]['area'] += area_coverage
        city_stats[city]['files'] += 1
        city_stats[city]['file_list'].append(pred_file)
        
        results.append({
            'file': pred_file,
            'city': city,
            'trees': num_trees,
            'coverage': area_coverage
        })

# Print overall summary statistics
print("\n" + "="*60)
print("=== OVERALL SUMMARY STATISTICS ===")
print("="*60)
print(f"Total number of trees detected: {total_trees:,}")
print(f"Average trees per image: {total_trees/len(pred_files):,.1f}")
print(f"Average tree coverage: {total_area/len(pred_files):.2f}%")

# Print city-wise statistics
print("\n" + "="*60)
print("=== TREE COUNT BY CITY ===")
print("="*60)

# Sort cities by tree count
sorted_cities = sorted(city_stats.items(), key=lambda x: x[1]['trees'], reverse=True)

for city, stats in sorted_cities:
    print(f"\n{city}:")
    print(f"  Total trees: {stats['trees']:,}")
    print(f"  Number of image tiles: {stats['files']}")
    print(f"  Average trees per tile: {stats['trees']/stats['files']:,.1f}")
    print(f"  Average coverage per tile: {stats['area']/stats['files']:.2f}%")

# Find images with most and least trees
max_trees = max(results, key=lambda x: x['trees'])
min_trees = min(results, key=lambda x: x['trees'])

print("\n" + "="*60)
print("=== EXTREMES ===")
print("="*60)
print(f"\nImage with most trees: {max_trees['file']}")
print(f"City: {max_trees['city']}")
print(f"Number of trees: {max_trees['trees']:,}")

print(f"\nImage with least trees: {min_trees['file']}")
print(f"City: {min_trees['city']}")
print(f"Number of trees: {min_trees['trees']:,}")

#Visuals

# Display small 256x256 image predictions from each city (PAN, NDVI, and Prediction)

def get_city_from_filename(filename):
    """Extract city name from prediction filename"""
    known_cities = ['atlanta', 'austin', 'bloomington', 'cupertino', 'surrey']
    filename_lower = filename.lower()
    for city in known_cities:
        if city in filename_lower:
            return city.title()
    return "Unknown"

def extract_256x256_sample(image_array, center=True):
    """
    Extract a 256x256 sample from an image array.
    If center=True, extracts from center. Otherwise extracts from top-left.
    """
    h, w = image_array.shape
    
    if h >= 256 and w >= 256:
        if center:
            # Extract from center
            start_h = (h - 256) // 2
            start_w = (w - 256) // 2
        else:
            # Extract from top-left
            start_h = 0
            start_w = 0
        return image_array[start_h:start_h+256, start_w:start_w+256]
    else:
        # If image is smaller than 256x256, pad it
        padded = np.zeros((256, 256), dtype=image_array.dtype)
        padded[:h, :w] = image_array
        return padded

def is_image_blank(image_array, threshold=0.001):
    """
    Check if an image is mostly blank (all zeros or very low variance)
    Returns True if image appears blank
    """
    if image_array.size == 0:
        return True
    
    # Check if all values are the same (zero variance)
    if np.std(image_array) < threshold:
        return True
    
    # Check if all values are very close to zero
    if np.max(np.abs(image_array)) < threshold:
        return True
    
    return False

def calculate_image_quality(pan_sample, ndvi_sample, pred_sample):
    """
    Calculate quality metrics for image samples
    Returns a score (higher = better) and metrics dictionary
    """
    from skimage import measure
    
    metrics = {}
    
    # 1. Tree density (prefer moderate: 10-30%)
    tree_mask = pred_sample > 0.5
    tree_density = np.sum(tree_mask) / pred_sample.size
    metrics['tree_density'] = tree_density
    
    # 2. Fragmentation: count connected components (fewer = less fragmented = better)
    labeled_array, num_components = measure.label(tree_mask, return_num=True, connectivity=2)
    metrics['num_components'] = num_components
    
    # 3. Average component size (larger = more coherent = better)
    if num_components > 0:
        component_sizes = [np.sum(labeled_array == i) for i in range(1, num_components + 1)]
        avg_component_size = np.mean(component_sizes)
        metrics['avg_component_size'] = avg_component_size
    else:
        metrics['avg_component_size'] = 0
    
    # 4. PAN image clarity: standard deviation (higher = more contrast = clearer)
    pan_std = np.std(pan_sample)
    metrics['pan_clarity'] = pan_std
    
    # 5. NDVI image clarity: standard deviation
    ndvi_std = np.std(ndvi_sample)
    metrics['ndvi_clarity'] = ndvi_std
    
    # Calculate composite quality score
    # Prefer: moderate density (10-30%), fewer components, larger components, clearer images
    density_score = 1.0 - abs(tree_density - 0.2) / 0.2  # Best at 20%
    fragmentation_score = 1.0 / (1.0 + num_components / 10.0)  # Penalize many components
    coherence_score = min(1.0, metrics['avg_component_size'] / 100.0)  # Prefer larger components
    clarity_score = min(1.0, (pan_std + ndvi_std) / 2.0)  # Prefer clearer images
    
    # Composite score (weighted)
    quality_score = (
        0.3 * density_score +
        0.3 * fragmentation_score +
        0.2 * coherence_score +
        0.2 * clarity_score
    )
    
    metrics['quality_score'] = quality_score
    
    return quality_score, metrics

def find_non_blank_file_pair(cutouts_dir, city_files, max_tries=30, max_tree_density=0.4, min_quality=0.3):
    """
    Find a non-blank set of PAN, NDVI, and prediction files for a city
    Ensures files are from the same geographic location by checking coordinates
    Prefers images with moderate tree density, low fragmentation, and good clarity
    Returns (pan_file, ndvi_file, pred_file) or None if all are blank/dense/low-quality
    
    Parameters:
    - max_tree_density: Maximum fraction of pixels that should be trees (default 0.4 = 40%)
    - min_quality: Minimum quality score threshold (default 0.3)
    """
    candidates = []
    
    for pred_file in city_files[:max_tries]:
        try:
            # Construct corresponding PAN and NDVI filenames
            # Handle different filename patterns
            if 'pred_pan_' in pred_file:
                # Format: pred_pan_XXX -> pan_XXX, ndvi_XXX
                pan_file = pred_file.replace('pred_', '')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
            elif 'pred_ndvi_' in pred_file:
                # Format: pred_ndvi_XXX -> ndvi_XXX, pan_XXX
                ndvi_file = pred_file.replace('pred_', '')
                pan_file = ndvi_file.replace('ndvi_', 'pan_')
            else:
                # Try to extract base name - handle various patterns
                base_name = pred_file.replace('pred_', '').replace('.tif', '')
                # Try different naming conventions
                if base_name.startswith('pan_'):
                    pan_file = base_name + '.tif'
                    ndvi_file = base_name.replace('pan_', 'ndvi_') + '.tif'
                elif base_name.startswith('ndvi_'):
                    ndvi_file = base_name + '.tif'
                    pan_file = base_name.replace('ndvi_', 'pan_') + '.tif'
                else:
                    # Assume base_name is the coordinate/identifier part
                    pan_file = f"pan_{base_name}.tif"
                    ndvi_file = f"ndvi_{base_name}.tif"
            
            pan_path = os.path.join(cutouts_dir, pan_file)
            ndvi_path = os.path.join(cutouts_dir, ndvi_file)
            pred_path = os.path.join(cutouts_dir, pred_file)
            
            # Check if all files exist
            if not (os.path.exists(pan_path) and os.path.exists(ndvi_path) and os.path.exists(pred_path)):
                continue
            
            # Load and verify files are from the same location by checking bounds
            with rasterio.open(pan_path) as pan_src:
                pan_bounds = pan_src.bounds
                pan_img = pan_src.read(1)
            with rasterio.open(ndvi_path) as ndvi_src:
                ndvi_bounds = ndvi_src.bounds
                ndvi_img = ndvi_src.read(1)
            with rasterio.open(pred_path) as pred_src:
                pred_bounds = pred_src.bounds
                pred_img = pred_src.read(1)
            
            # Verify all files have the same bounds (same geographic location)
            # Allow small tolerance for floating point differences
            bounds_tolerance = 0.001  # ~100 meters
            
            pan_center = ((pan_bounds.left + pan_bounds.right) / 2, (pan_bounds.bottom + pan_bounds.top) / 2)
            ndvi_center = ((ndvi_bounds.left + ndvi_bounds.right) / 2, (ndvi_bounds.bottom + ndvi_bounds.top) / 2)
            pred_center = ((pred_bounds.left + pred_bounds.right) / 2, (pred_bounds.bottom + pred_bounds.top) / 2)
            
            # Check if centers are close enough (same location)
            pan_ndvi_dist = np.sqrt((pan_center[0] - ndvi_center[0])**2 + (pan_center[1] - ndvi_center[1])**2)
            pan_pred_dist = np.sqrt((pan_center[0] - pred_center[0])**2 + (pan_center[1] - pred_center[1])**2)
            
            if pan_ndvi_dist > bounds_tolerance or pan_pred_dist > bounds_tolerance:
                # Files are not from the same location, skip
                continue
            
            # Check if any of them are blank
            if is_image_blank(pan_img) or is_image_blank(ndvi_img) or is_image_blank(pred_img):
                continue
            
            # Extract samples from the same relative position (center)
            pan_sample = extract_256x256_sample(pan_img, center=True)
            ndvi_sample = extract_256x256_sample(ndvi_img, center=True)
            pred_sample = extract_256x256_sample(pred_img, center=True)
            
            if is_image_blank(pan_sample) or is_image_blank(ndvi_sample) or is_image_blank(pred_sample):
                continue
            
            # Check tree density in prediction sample
            tree_mask = pred_sample > 0.5
            tree_density = np.sum(tree_mask) / pred_sample.size
            
            # Skip if too dense with trees
            if tree_density > max_tree_density:
                continue
            
            # Calculate image quality metrics
            quality_score, metrics = calculate_image_quality(pan_sample, ndvi_sample, pred_sample)
            
            # Skip if quality is too low
            if quality_score < min_quality:
                continue
            
            # Store candidate with quality metrics
            candidates.append((pan_file, ndvi_file, pred_file, tree_density, pan_center, quality_score, metrics))
                
        except Exception as e:
            continue
    
    # If we have candidates, prefer ones with highest quality score
    if candidates:
        # Sort by quality score (highest first)
        candidates.sort(key=lambda x: x[5], reverse=True)
        return candidates[0][:3]  # Return (pan_file, ndvi_file, pred_file)
    
    return None

def load_city_boundaries(area_dir='Preprocessing/input/area'):
    """Load all city boundary shapefiles"""
    import geopandas as gpd
    import pandas as pd
    
    city_boundaries = {}
    
    if not os.path.exists(area_dir):
        print(f"Warning: Area directory {area_dir} not found")
        return city_boundaries
    
    shp_files = [f for f in os.listdir(area_dir) if f.endswith('.shp')]
    
    for shp_file in shp_files:
        city_name = shp_file.split('_')[0].title()
        shp_path = os.path.join(area_dir, shp_file)
        
        try:
            gdf = gpd.read_file(shp_path)
            if city_name not in city_boundaries:
                city_boundaries[city_name] = []
            city_boundaries[city_name].append(gdf)
        except Exception as e:
            print(f"Error loading {shp_file}: {e}")
    
    # Merge multiple boundaries for the same city
    for city in list(city_boundaries.keys()):
        if len(city_boundaries[city]) > 1:
            city_boundaries[city] = pd.concat(city_boundaries[city], ignore_index=True)
        else:
            city_boundaries[city] = city_boundaries[city][0]
    
    return city_boundaries

def crop_to_city_boundary(img, transform, crs, city_boundary_gdf):
    """
    Crop image to city boundary using rasterio mask
    Returns cropped image and its transform
    """
    from rasterio.mask import mask
    from rasterio.io import MemoryFile
    
    if city_boundary_gdf is None:
        return img, transform
    
    try:
        # Ensure boundary is in the same CRS as the image
        if city_boundary_gdf.crs != crs:
            city_boundary_gdf = city_boundary_gdf.to_crs(crs)
        
        # Get geometries from GeoDataFrame
        import geopandas as gpd
        if isinstance(city_boundary_gdf, gpd.GeoDataFrame):
            # It's a GeoDataFrame - get all geometries
            geometries = city_boundary_gdf.geometry.tolist()
        elif hasattr(city_boundary_gdf, 'geometry'):
            # It might be a Series or similar
            geometries = [city_boundary_gdf.geometry] if hasattr(city_boundary_gdf.geometry, '__geo_interface__') else city_boundary_gdf.geometry.tolist()
        else:
            # Assume it's already a list of geometries
            geometries = city_boundary_gdf if isinstance(city_boundary_gdf, list) else [city_boundary_gdf]
        
        # Create an in-memory dataset for masking
        # We need to create a proper dataset object, not just a dict
        profile = {
            'driver': 'GTiff',
            'dtype': img.dtype,
            'count': 1,
            'width': img.shape[1],
            'height': img.shape[0],
            'transform': transform,
            'crs': crs
        }
        
        # Create a temporary in-memory file
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(img, 1)
                
                # Now crop using mask with the dataset object
                out_image, out_transform = mask(
                    dataset,
                    geometries,
                    crop=True,
                    filled=False,
                    nodata=np.nan
                )
        
        # Return the masked array (remove the extra dimension)
        if len(out_image.shape) == 3:
            return out_image[0], out_transform
        return out_image, out_transform
        
    except Exception as e:
        print(f"Warning: Could not crop to boundary: {e}")
        import traceback
        traceback.print_exc()
        return img, transform

def display_city_predictions(cutouts_dir=None, figsize=None, crop_to_boundaries=True):
    """
    Display PAN, NDVI, and Prediction images (full images) from each city
    Optionally crops images to city boundaries
    
    Parameters:
    - crop_to_boundaries: If True, crop images to city boundary shapefiles
    """
    # Load city boundaries if cropping is requested
    city_boundaries = {}
    if crop_to_boundaries:
        try:
            city_boundaries = load_city_boundaries('Preprocessing/input/area')
            if city_boundaries:
                print(f"Loaded city boundaries for: {', '.join(city_boundaries.keys())}")
            else:
                print("Warning: No city boundaries found, displaying full images")
                crop_to_boundaries = False
        except Exception as e:
            print(f"Warning: Could not load city boundaries: {e}")
            crop_to_boundaries = False
    
    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])
    
    if not pred_files:
        print(f"No prediction files found in {cutouts_dir}")
        return
    
    # Group files by city
    city_files = {}
    for pred_file in pred_files:
        city = get_city_from_filename(pred_file)
        if city not in city_files:
            city_files[city] = []
        city_files[city].append(pred_file)
    
    # Sort cities alphabetically
    sorted_cities = sorted(city_files.keys())
    
    # Create figure: 3 columns (PAN, NDVI, Prediction) for each city
    n_cities = len(sorted_cities)
    
    # Auto-calculate figure size based on number of cities and image sizes
    if figsize is None:
        # Larger figure size for full images - scale based on typical image dimensions
        # Estimate: assume images are ~5000px wide, want ~6 inches per image
        figsize = (24, 8 * n_cities)
    
    # Set higher DPI for better resolution
    fig, axes = plt.subplots(n_cities, 3, figsize=figsize, dpi=100)
    
    # Handle single city case
    if n_cities == 1:
        axes = axes.reshape(1, -1)
    
    # Add column headers
    column_labels = ['PAN Image', 'NDVI Image', 'Prediction']
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes, 
                             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    print(f"Found {len(pred_files)} prediction files across {n_cities} cities")
    if crop_to_boundaries:
        print(f"Displaying PAN, NDVI, and Prediction images cropped to city boundaries:\n")
    else:
        print(f"Displaying full PAN, NDVI, and Prediction images from each city:\n")
    
    for city_idx, city in enumerate(sorted_cities):
        # Find non-blank file set for this city
        file_set = find_non_blank_file_pair(cutouts_dir, city_files[city])
        
        if file_set is None:
            # If no good files found, try the first one anyway
            pred_file = city_files[city][0]
            if 'pred_pan_' in pred_file:
                pan_file = pred_file.replace('pred_', '')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
            else:
                base_name = pred_file.replace('pred_', '').replace('.tif', '')
                pan_file = f"pan_{base_name}.tif"
                ndvi_file = f"ndvi_{base_name}.tif"
            file_set = (pan_file, ndvi_file, pred_file)
        
        pan_file, ndvi_file, pred_file = file_set
        pan_path = os.path.join(cutouts_dir, pan_file)
        ndvi_path = os.path.join(cutouts_dir, ndvi_file)
        pred_path = os.path.join(cutouts_dir, pred_file)
        
        try:
            # Load full images with metadata
            with rasterio.open(pan_path) as src:
                pan_img = src.read(1)
                pan_shape = pan_img.shape
                pan_total_pixels = pan_img.size
                pan_transform = src.transform
                pan_crs = src.crs
            with rasterio.open(ndvi_path) as src:
                ndvi_img = src.read(1)
                ndvi_shape = ndvi_img.shape
                ndvi_total_pixels = ndvi_img.size
                ndvi_transform = src.transform
                ndvi_crs = src.crs
            with rasterio.open(pred_path) as src:
                pred_img = src.read(1)
                pred_shape = pred_img.shape
                pred_total_pixels = pred_img.size
                pred_transform = src.transform
                pred_crs = src.crs
            
            # Crop to city boundary if requested and boundary exists
            city_boundary = city_boundaries.get(city) if crop_to_boundaries else None
            
            if city_boundary is not None:
                pan_img, pan_transform = crop_to_city_boundary(pan_img, pan_transform, pan_crs, city_boundary)
                ndvi_img, ndvi_transform = crop_to_city_boundary(ndvi_img, ndvi_transform, ndvi_crs, city_boundary)
                pred_img, pred_transform = crop_to_city_boundary(pred_img, pred_transform, pred_crs, city_boundary)
                
                pan_shape = pan_img.shape
                ndvi_shape = ndvi_img.shape
                pred_shape = pred_img.shape
                print(f"{city}: Cropped to city boundary - PAN: {pan_shape[0]}x{pan_shape[1]}, "
                      f"NDVI: {ndvi_shape[0]}x{ndvi_shape[1]}, Prediction: {pred_shape[0]}x{pred_shape[1]}")
            else:
                print(f"{city}: Loading full images - PAN: {pan_shape[0]}x{pan_shape[1]} ({pan_total_pixels:,} pixels), "
                      f"NDVI: {ndvi_shape[0]}x{ndvi_shape[1]} ({ndvi_total_pixels:,} pixels), "
                      f"Prediction: {pred_shape[0]}x{pred_shape[1]} ({pred_total_pixels:,} pixels)")
            
            # Display full images without any downsampling or interpolation
            # Use interpolation='nearest' to preserve full pixel resolution
            # Set rasterized=False to ensure full resolution is displayed
            
            # Calculate quality metrics on center sample (for stats)
            pan_sample = extract_256x256_sample(pan_img, center=True)
            ndvi_sample = extract_256x256_sample(ndvi_img, center=True)
            pred_sample = extract_256x256_sample(pred_img, center=True)
            quality_score, quality_metrics = calculate_image_quality(pan_sample, ndvi_sample, pred_sample)
            
            # Display PAN image - use 'Greys' colormap matching reference image
            pan_ax = axes[city_idx, 0]
            pan_min, pan_max = pan_img.min(), pan_img.max()
            im1 = pan_ax.imshow(pan_img, cmap='Greys', vmin=pan_min, vmax=pan_max, 
                               aspect='auto', interpolation='nearest', rasterized=False)
            title_suffix = " (cropped to boundary)" if city_boundary is not None else ""
            pan_ax.set_title(f'{city} - Full Image ({pan_shape[0]}x{pan_shape[1]}){title_suffix}', 
                           fontsize=12, fontweight='bold', pad=10)
            pan_ax.axis('off')
            plt.colorbar(im1, ax=pan_ax, fraction=0.046, pad=0.04)
            
            # Display NDVI image - RdYlGn gives green-yellow-red appearance matching reference
            ndvi_ax = axes[city_idx, 1]
            ndvi_clipped = np.clip(ndvi_img, -1, 1)
            im2 = ndvi_ax.imshow(ndvi_clipped, cmap='RdYlGn', vmin=-1, vmax=1, 
                                aspect='auto', interpolation='nearest', rasterized=False)
            ndvi_ax.set_title(f'{city} - Full Image ({ndvi_shape[0]}x{ndvi_shape[1]}){title_suffix}', 
                            fontsize=12, fontweight='bold', pad=10)
            ndvi_ax.axis('off')
            plt.colorbar(im2, ax=ndvi_ax, fraction=0.046, pad=0.04)
            
            # Display Prediction
            pred_ax = axes[city_idx, 2]
            im3 = pred_ax.imshow(pred_img, cmap='viridis', vmin=0, vmax=1, 
                                aspect='auto', interpolation='nearest', rasterized=False)
            pred_ax.set_title(f'{city} - Full Image ({pred_shape[0]}x{pred_shape[1]}){title_suffix}', 
                            fontsize=12, fontweight='bold', pad=10)
            pred_ax.axis('off')
            plt.colorbar(im3, ax=pred_ax, fraction=0.046, pad=0.04)
            
            # Calculate tree density for full image
            tree_mask_full = pred_img > 0.5
            tree_density_full = np.sum(tree_mask_full) / pred_img.size
            
            # Print stats including quality metrics
            print(f"{city}:")
            print(f"  Files: {pan_file[:40]}... / {ndvi_file[:40]}... / {pred_file[:40]}...")
            print(f"  Image sizes: PAN={pan_shape[0]}x{pan_shape[1]}, NDVI={ndvi_shape[0]}x{ndvi_shape[1]}, Prediction={pred_shape[0]}x{pred_shape[1]}")
            print(f"  Quality Score: {quality_score:.3f} (based on center sample)")
            print(f"    - Tree density (full image): {tree_density_full*100:.1f}%")
            print(f"    - Connected components: {quality_metrics['num_components']} (fewer = less fragmented)")
            print(f"    - Avg component size: {quality_metrics['avg_component_size']:.1f} pixels")
            print(f"    - Image clarity: {quality_metrics['pan_clarity']:.3f} (PAN), {quality_metrics['ndvi_clarity']:.3f} (NDVI)")
            print()
            
        except Exception as e:
            # Show error message
            for col_idx in range(3):
                axes[city_idx, col_idx].text(0.5, 0.5, f'Error\nloading\n{city}', 
                          ha='center', va='center', transform=axes[city_idx, col_idx].transAxes)
                axes[city_idx, col_idx].set_title(f'{city}', fontsize=12)
            print(f"Error loading {city}: {e}\n")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for column headers
    # Save figure instead of showing (for batch processing)
    output_path = os.path.join(cutouts_dir, 'city_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return city_files
    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])
    
    if not pred_files:
        print(f"No prediction files found in {cutouts_dir}")
        return
    
    # Group files by city
    city_files = {}
    for pred_file in pred_files:
        city = get_city_from_filename(pred_file)
        if city not in city_files:
            city_files[city] = []
        city_files[city].append(pred_file)
    
    # Sort cities alphabetically
    sorted_cities = sorted(city_files.keys())
    
    # Create figure: 3 columns (PAN, NDVI, Prediction) for each city
    n_cities = len(sorted_cities)
    
    # Auto-calculate figure size based on number of cities and image sizes
    if figsize is None:
        # Larger figure size for full images - scale based on typical image dimensions
        # Estimate: assume images are ~5000px wide, want ~6 inches per image
        figsize = (24, 8 * n_cities)
    
    # Set higher DPI for better resolution
    fig, axes = plt.subplots(n_cities, 3, figsize=figsize, dpi=100)
    
    # Handle single city case
    if n_cities == 1:
        axes = axes.reshape(1, -1)
    
    # Add column headers
    column_labels = ['PAN Image', 'NDVI Image', 'Prediction']
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes, 
                             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    print(f"Found {len(pred_files)} prediction files across {n_cities} cities")
    print(f"Displaying full PAN, NDVI, and Prediction images from each city:\n")
    
    for city_idx, city in enumerate(sorted_cities):
        # Find non-blank file set for this city
        file_set = find_non_blank_file_pair(cutouts_dir, city_files[city])
        
        if file_set is None:
            # If no good files found, try the first one anyway
            pred_file = city_files[city][0]
            if 'pred_pan_' in pred_file:
                pan_file = pred_file.replace('pred_', '')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
            else:
                base_name = pred_file.replace('pred_', '').replace('.tif', '')
                pan_file = f"pan_{base_name}.tif"
                ndvi_file = f"ndvi_{base_name}.tif"
            file_set = (pan_file, ndvi_file, pred_file)
        
        pan_file, ndvi_file, pred_file = file_set
        pan_path = os.path.join(cutouts_dir, pan_file)
        ndvi_path = os.path.join(cutouts_dir, ndvi_file)
        pred_path = os.path.join(cutouts_dir, pred_file)
        
        try:
            # Load full images - verify we're getting complete images
            with rasterio.open(pan_path) as src:
                pan_img = src.read(1)
                pan_shape = pan_img.shape
                pan_total_pixels = pan_img.size
            with rasterio.open(ndvi_path) as src:
                ndvi_img = src.read(1)
                ndvi_shape = ndvi_img.shape
                ndvi_total_pixels = ndvi_img.size
            with rasterio.open(pred_path) as src:
                pred_img = src.read(1)
                pred_shape = pred_img.shape
                pred_total_pixels = pred_img.size
            
            # Verify images are full size (not samples)
            print(f"{city}: Loading full images - PAN: {pan_shape[0]}x{pan_shape[1]} ({pan_total_pixels:,} pixels), "
                  f"NDVI: {ndvi_shape[0]}x{ndvi_shape[1]} ({ndvi_total_pixels:,} pixels), "
                  f"Prediction: {pred_shape[0]}x{pred_shape[1]} ({pred_total_pixels:,} pixels)")
            
            # Display full images without any downsampling or interpolation
            # Use interpolation='nearest' to preserve full pixel resolution
            # Set rasterized=False to ensure full resolution is displayed
            
            # Calculate quality metrics on center sample (for stats)
            pan_sample = extract_256x256_sample(pan_img, center=True)
            ndvi_sample = extract_256x256_sample(ndvi_img, center=True)
            pred_sample = extract_256x256_sample(pred_img, center=True)
            quality_score, quality_metrics = calculate_image_quality(pan_sample, ndvi_sample, pred_sample)
            
            # Display PAN image - use 'Greys' colormap matching reference image
            pan_ax = axes[city_idx, 0]
            pan_min, pan_max = pan_img.min(), pan_img.max()
            im1 = pan_ax.imshow(pan_img, cmap='Greys', vmin=pan_min, vmax=pan_max, 
                               aspect='auto', interpolation='nearest', rasterized=False)
            pan_ax.set_title(f'{city} - Full Image ({pan_shape[0]}x{pan_shape[1]})', fontsize=12, fontweight='bold', pad=10)
            pan_ax.axis('off')
            plt.colorbar(im1, ax=pan_ax, fraction=0.046, pad=0.04)
            
            # Display NDVI image - RdYlGn gives green-yellow-red appearance matching reference
            ndvi_ax = axes[city_idx, 1]
            ndvi_clipped = np.clip(ndvi_img, -1, 1)
            im2 = ndvi_ax.imshow(ndvi_clipped, cmap='RdYlGn', vmin=-1, vmax=1, 
                                aspect='auto', interpolation='nearest', rasterized=False)
            ndvi_ax.set_title(f'{city} - Full Image ({ndvi_shape[0]}x{ndvi_shape[1]})', fontsize=12, fontweight='bold', pad=10)
            ndvi_ax.axis('off')
            plt.colorbar(im2, ax=ndvi_ax, fraction=0.046, pad=0.04)
            
            # Display Prediction
            pred_ax = axes[city_idx, 2]
            im3 = pred_ax.imshow(pred_img, cmap='viridis', vmin=0, vmax=1, 
                                aspect='auto', interpolation='nearest', rasterized=False)
            pred_ax.set_title(f'{city} - Full Image ({pred_shape[0]}x{pred_shape[1]})', fontsize=12, fontweight='bold', pad=10)
            pred_ax.axis('off')
            plt.colorbar(im3, ax=pred_ax, fraction=0.046, pad=0.04)
            
            # Calculate tree density for full image
            tree_mask_full = pred_img > 0.5
            tree_density_full = np.sum(tree_mask_full) / pred_img.size
            
            # Print stats including quality metrics
            print(f"{city}:")
            print(f"  Files: {pan_file[:40]}... / {ndvi_file[:40]}... / {pred_file[:40]}...")
            print(f"  Full image sizes: PAN={pan_shape[0]}x{pan_shape[1]}, NDVI={ndvi_shape[0]}x{ndvi_shape[1]}, Prediction={pred_shape[0]}x{pred_shape[1]}")
            print(f"  Quality Score: {quality_score:.3f} (based on center sample)")
            print(f"    - Tree density (full image): {tree_density_full*100:.1f}%")
            print(f"    - Connected components: {quality_metrics['num_components']} (fewer = less fragmented)")
            print(f"    - Avg component size: {quality_metrics['avg_component_size']:.1f} pixels")
            print(f"    - Image clarity: {quality_metrics['pan_clarity']:.3f} (PAN), {quality_metrics['ndvi_clarity']:.3f} (NDVI)")
            print()
            
        except Exception as e:
            # Show error message
            for col_idx in range(3):
                axes[city_idx, col_idx].text(0.5, 0.5, f'Error\nloading\n{city}', 
                          ha='center', va='center', transform=axes[city_idx, col_idx].transAxes)
                axes[city_idx, col_idx].set_title(f'{city}', fontsize=12)
            print(f"Error loading {city}: {e}\n")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for column headers
    # Save figure instead of showing (for batch processing)
    output_path = os.path.join(cutouts_dir, 'city_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return city_files
    
    for city_idx, city in enumerate(sorted_cities):
        # Find non-blank file set for this city
        file_set = find_non_blank_file_pair(cutouts_dir, city_files[city])
        
        if file_set is None:
            # If no good files found, try the first one anyway
            pred_file = city_files[city][0]
            if 'pred_pan_' in pred_file:
                pan_file = pred_file.replace('pred_', '')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
            else:
                base_name = pred_file.replace('pred_', '').replace('.tif', '')
                pan_file = f"pan_{base_name}.tif"
                ndvi_file = f"ndvi_{base_name}.tif"
            file_set = (pan_file, ndvi_file, pred_file)
        
        pan_file, ndvi_file, pred_file = file_set
        pan_path = os.path.join(cutouts_dir, pan_file)
        ndvi_path = os.path.join(cutouts_dir, ndvi_file)
        pred_path = os.path.join(cutouts_dir, pred_file)
        
        try:
            # Load images
            with rasterio.open(pan_path) as src:
                pan_img = src.read(1)
            with rasterio.open(ndvi_path) as src:
                ndvi_img = src.read(1)
            with rasterio.open(pred_path) as src:
                pred_img = src.read(1)
            
            # Extract 256x256 samples
            pan_sample = extract_256x256_sample(pan_img, center=True)
            ndvi_sample = extract_256x256_sample(ndvi_img, center=True)
            pred_sample = extract_256x256_sample(pred_img, center=True)
            
            # Display PAN image
            pan_ax = axes[city_idx, 0]
            im1 = pan_ax.imshow(pan_sample, cmap='gray')
            pan_ax.set_title(f'{city}', fontsize=12, fontweight='bold', pad=10)
            pan_ax.axis('off')
            plt.colorbar(im1, ax=pan_ax, fraction=0.046, pad=0.04)
            
            # Display NDVI image (clip extreme values for visualization)
            ndvi_ax = axes[city_idx, 1]
            ndvi_clipped = np.clip(ndvi_sample, -1, 1)
            im2 = ndvi_ax.imshow(ndvi_clipped, cmap='RdYlGn', vmin=-1, vmax=1)
            ndvi_ax.set_title(f'{city}', fontsize=12, fontweight='bold', pad=10)
            ndvi_ax.axis('off')
            plt.colorbar(im2, ax=ndvi_ax, fraction=0.046, pad=0.04)
            
            # Display Prediction
            pred_ax = axes[city_idx, 2]
            im3 = pred_ax.imshow(pred_sample, cmap='viridis', vmin=0, vmax=1)
            pred_ax.set_title(f'{city}', fontsize=12, fontweight='bold', pad=10)
            pred_ax.axis('off')
            plt.colorbar(im3, ax=pred_ax, fraction=0.046, pad=0.04)
            
            # Print stats
            print(f"{city}:")
            print(f"  Files: {pan_file[:40]}... / {ndvi_file[:40]}... / {pred_file[:40]}...")
            print(f"  PAN sample - Range: [{pan_sample.min():.3f}, {pan_sample.max():.3f}], Mean: {pan_sample.mean():.3f}, Std: {pan_sample.std():.3f}")
            print(f"  NDVI sample - Range: [{ndvi_sample.min():.3f}, {ndvi_sample.max():.3f}], Mean: {ndvi_sample.mean():.3f}, Std: {ndvi_sample.std():.3f}")
            print(f"  Prediction sample - Range: [{pred_sample.min():.3f}, {pred_sample.max():.3f}], Mean: {pred_sample.mean():.3f}, Std: {pred_sample.std():.3f}")
            print()
            
        except Exception as e:
            # Show error message
            for col_idx in range(3):
                axes[city_idx, col_idx].text(0.5, 0.5, f'Error\nloading\n{city}', 
                          ha='center', va='center', transform=axes[city_idx, col_idx].transAxes)
                axes[city_idx, col_idx].set_title(f'{city}', fontsize=12)
            print(f"Error loading {city}: {e}\n")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for column headers
    # Save figure instead of showing (for batch processing)
    output_path = os.path.join(cutouts_dir, 'city_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return city_files
    
    # Handle single city case
    if n_cities == 1:
        axes = axes.reshape(1, -1)
    
    # Add column headers
    column_labels = ['PAN Image', 'NDVI Image', 'Prediction']
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes, 
                             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    print(f"Found {len(pred_files)} prediction files across {n_cities} cities")
    print(f"Displaying PAN, NDVI, and Prediction (256x256 samples) from each city:\n")
    
    for city_idx, city in enumerate(sorted_cities):
        # Find non-blank file set for this city
        file_set = find_non_blank_file_pair(cutouts_dir, city_files[city])
        
        if file_set is None:
            # If no good files found, try the first one anyway
            pred_file = city_files[city][0]
            if 'pred_pan_' in pred_file:
                pan_file = pred_file.replace('pred_', '')
                ndvi_file = pan_file.replace('pan_', 'ndvi_')
            else:
                base_name = pred_file.replace('pred_', '').replace('.tif', '')
                pan_file = f"pan_{base_name}.tif"
                ndvi_file = f"ndvi_{base_name}.tif"
            file_set = (pan_file, ndvi_file, pred_file)
        
        pan_file, ndvi_file, pred_file = file_set
        pan_path = os.path.join(cutouts_dir, pan_file)
        ndvi_path = os.path.join(cutouts_dir, ndvi_file)
        pred_path = os.path.join(cutouts_dir, pred_file)
        
        try:
            # Load images
            with rasterio.open(pan_path) as src:
                pan_img = src.read(1)
            with rasterio.open(ndvi_path) as src:
                ndvi_img = src.read(1)
            with rasterio.open(pred_path) as src:
                pred_img = src.read(1)
            
            # Extract 256x256 samples
            pan_sample = extract_256x256_sample(pan_img, center=True)
            ndvi_sample = extract_256x256_sample(ndvi_img, center=True)
            pred_sample = extract_256x256_sample(pred_img, center=True)
            
            # Display PAN image
            pan_ax = axes[city_idx, 0]
            im1 = pan_ax.imshow(pan_sample, cmap='gray')
            pan_ax.set_title(f'{city}', fontsize=12, fontweight='bold', pad=10)
            pan_ax.axis('off')
            plt.colorbar(im1, ax=pan_ax, fraction=0.046, pad=0.04)
            
            # Display NDVI image (clip extreme values for visualization)
            ndvi_ax = axes[city_idx, 1]
            ndvi_clipped = np.clip(ndvi_sample, -1, 1)
            im2 = ndvi_ax.imshow(ndvi_clipped, cmap='RdYlGn', vmin=-1, vmax=1)
            ndvi_ax.set_title(f'{city}', fontsize=12, fontweight='bold', pad=10)
            ndvi_ax.axis('off')
            plt.colorbar(im2, ax=ndvi_ax, fraction=0.046, pad=0.04)
            
            # Display Prediction
            pred_ax = axes[city_idx, 2]
            im3 = pred_ax.imshow(pred_sample, cmap='viridis', vmin=0, vmax=1)
            pred_ax.set_title(f'{city}', fontsize=12, fontweight='bold', pad=10)
            pred_ax.axis('off')
            plt.colorbar(im3, ax=pred_ax, fraction=0.046, pad=0.04)
            
            # Print stats
            print(f"{city}:")
            print(f"  Files: {pan_file[:40]}... / {ndvi_file[:40]}... / {pred_file[:40]}...")
            print(f"  PAN sample - Range: [{pan_sample.min():.3f}, {pan_sample.max():.3f}], Mean: {pan_sample.mean():.3f}, Std: {pan_sample.std():.3f}")
            print(f"  NDVI sample - Range: [{ndvi_sample.min():.3f}, {ndvi_sample.max():.3f}], Mean: {ndvi_sample.mean():.3f}, Std: {ndvi_sample.std():.3f}")
            print(f"  Prediction sample - Range: [{pred_sample.min():.3f}, {pred_sample.max():.3f}], Mean: {pred_sample.mean():.3f}, Std: {pred_sample.std():.3f}")
            print()
            
        except Exception as e:
            # Show error message
            for col_idx in range(3):
                axes[city_idx, col_idx].text(0.5, 0.5, f'Error\nloading\n{city}', 
                          ha='center', va='center', transform=axes[city_idx, col_idx].transAxes)
                axes[city_idx, col_idx].set_title(f'{city}', fontsize=12)
            print(f"Error loading {city}: {e}\n")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for column headers
    # Save figure instead of showing (for batch processing)
    output_path = os.path.join(cutouts_dir, 'city_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return city_files

# Display random prediction image from each city with paired PAN and NDVI

def display_random_city_images(cutouts_dir="cutouts", figsize=None):
    """
    Randomly select one prediction image from each city and display 
    the paired PAN, NDVI, and Prediction images
    """
    import random
    
    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])
    
    if not pred_files:
        print(f"No prediction files found in {cutouts_dir}")
        return
    
    # Group files by city
    city_files = {}
    for pred_file in pred_files:
        city = get_city_from_filename(pred_file)
        if city not in city_files:
            city_files[city] = []
        city_files[city].append(pred_file)
    
    # Sort cities alphabetically
    sorted_cities = sorted(city_files.keys())
    
    # Randomly select one file per city
    selected_files = {}
    for city in sorted_cities:
        selected_pred = random.choice(city_files[city])
        selected_files[city] = selected_pred
    
    # Create figure: 3 columns (PAN, NDVI, Prediction) for each city
    n_cities = len(sorted_cities)
    
    if figsize is None:
        figsize = (24, 8 * n_cities)
    
    fig, axes = plt.subplots(n_cities, 3, figsize=figsize, dpi=100)
    
    # Handle single city case
    if n_cities == 1:
        axes = axes.reshape(1, -1)
    
    # Add column headers
    column_labels = ['PAN Image', 'NDVI Image', 'Prediction']
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes, 
                             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    print(f"Found {len(pred_files)} prediction files across {n_cities} cities")
    print(f"Randomly selected one image per city:\n")
    
    for city_idx, city in enumerate(sorted_cities):
        pred_file = selected_files[city]
        
        # Find corresponding PAN and NDVI files
        if 'pred_pan_' in pred_file:
            pan_file = pred_file.replace('pred_', '')
            ndvi_file = pan_file.replace('pan_', 'ndvi_')
        elif 'pred_ndvi_' in pred_file:
            ndvi_file = pred_file.replace('pred_', '')
            pan_file = ndvi_file.replace('ndvi_', 'pan_')
        else:
            base_name = pred_file.replace('pred_', '').replace('.tif', '')
            pan_file = f"pan_{base_name}.tif"
            ndvi_file = f"ndvi_{base_name}.tif"
        
        pan_path = os.path.join(cutouts_dir, pan_file)
        ndvi_path = os.path.join(cutouts_dir, ndvi_file)
        pred_path = os.path.join(cutouts_dir, pred_file)
        
        try:
            # Load images
            with rasterio.open(pan_path) as src:
                pan_img = src.read(1)
                pan_shape = pan_img.shape
            with rasterio.open(ndvi_path) as src:
                ndvi_img = src.read(1)
                ndvi_shape = ndvi_img.shape
            with rasterio.open(pred_path) as src:
                pred_img = src.read(1)
                pred_shape = pred_img.shape
            
            print(f"{city}: Selected {pred_file}")
            print(f"  PAN: {pan_file} ({pan_shape[0]}x{pan_shape[1]})")
            print(f"  NDVI: {ndvi_file} ({ndvi_shape[0]}x{ndvi_shape[1]})")
            print(f"  Prediction: {pred_file} ({pred_shape[0]}x{pred_shape[1]})")
            
            # Display PAN image
            pan_ax = axes[city_idx, 0]
            pan_min, pan_max = pan_img.min(), pan_img.max()
            im1 = pan_ax.imshow(pan_img, cmap='Greys', vmin=pan_min, vmax=pan_max, 
                               aspect='auto', interpolation='nearest', rasterized=False)
            pan_ax.set_title(f'{city} - PAN', fontsize=12, fontweight='bold', pad=10)
            pan_ax.axis('off')
            plt.colorbar(im1, ax=pan_ax, fraction=0.046, pad=0.04)
            
            # Display NDVI image
            ndvi_ax = axes[city_idx, 1]
            ndvi_clipped = np.clip(ndvi_img, -1, 1)
            im2 = ndvi_ax.imshow(ndvi_clipped, cmap='RdYlGn', vmin=-1, vmax=1, 
                                aspect='auto', interpolation='nearest', rasterized=False)
            ndvi_ax.set_title(f'{city} - NDVI', fontsize=12, fontweight='bold', pad=10)
            ndvi_ax.axis('off')
            plt.colorbar(im2, ax=ndvi_ax, fraction=0.046, pad=0.04)
            
            # Display Prediction
            pred_ax = axes[city_idx, 2]
            im3 = pred_ax.imshow(pred_img, cmap='viridis', vmin=0, vmax=1, 
                                aspect='auto', interpolation='nearest', rasterized=False)
            pred_ax.set_title(f'{city} - Prediction', fontsize=12, fontweight='bold', pad=10)
            pred_ax.axis('off')
            plt.colorbar(im3, ax=pred_ax, fraction=0.046, pad=0.04)
            
            print()
            
        except Exception as e:
            # Show error message
            for col_idx in range(3):
                axes[city_idx, col_idx].text(0.5, 0.5, f'Error\nloading\n{city}', 
                          ha='center', va='center', transform=axes[city_idx, col_idx].transAxes)
                axes[city_idx, col_idx].set_title(f'{city}', fontsize=12)
            print(f"Error loading {city}: {e}\n")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    # Save figure instead of showing (for batch processing)
    output_path = os.path.join(cutouts_dir, 'random_city_images.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return selected_files

# Display random images from each city (commented out for batch processing)
# Uncomment if you want to generate visualization images
# selected_files = display_random_city_images(cutouts_dir=CUTOUTS_DIR)

# Calculate % canopy cover for each city

def calculate_canopy_cover_by_city(cutouts_dir=None, threshold=0.5):
    """
    Calculate percentage canopy cover for each city
    
    Parameters:
    - threshold: Threshold value for tree detection (default 0.5)
    
    Returns:
    - Dictionary with city names as keys and canopy cover stats as values
    """
    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(cutouts_dir) if f.startswith('pred_') and f.endswith('.tif')])
    
    if not pred_files:
        print(f"No prediction files found in {cutouts_dir}")
        return {}
    
    # Group files by city
    city_files = {}
    for pred_file in pred_files:
        city = get_city_from_filename(pred_file)
        if city not in city_files:
            city_files[city] = []
        city_files[city].append(pred_file)
    
    # Sort cities alphabetically
    sorted_cities = sorted(city_files.keys())
    
    city_stats = {}
    
    print(f"Calculating canopy cover for {len(sorted_cities)} cities...")
    print(f"Using threshold: {threshold}\n")
    
    for city in sorted_cities:
        city_files_list = city_files[city]
        total_pixels = 0
        canopy_pixels = 0
        num_images = 0
        image_stats = []
        
        print(f"Processing {city} ({len(city_files_list)} images)...")
        
        for pred_file in tqdm(city_files_list, desc=f"{city}", leave=False):
            pred_path = os.path.join(cutouts_dir, pred_file)
            
            try:
                with rasterio.open(pred_path) as src:
                    pred_img = src.read(1)
                    
                    # Calculate canopy cover for this image
                    img_total_pixels = pred_img.size
                    img_canopy_mask = pred_img > threshold
                    img_canopy_pixels = np.sum(img_canopy_mask)
                    img_canopy_cover = (img_canopy_pixels / img_total_pixels) * 100
                    
                    total_pixels += img_total_pixels
                    canopy_pixels += img_canopy_pixels
                    num_images += 1
                    
                    image_stats.append({
                        'file': pred_file,
                        'canopy_cover': img_canopy_cover,
                        'total_pixels': img_total_pixels,
                        'canopy_pixels': img_canopy_pixels
                    })
                    
            except Exception as e:
                print(f"  Error processing {pred_file}: {e}")
                continue
        
        # Calculate overall canopy cover for the city
        if total_pixels > 0:
            overall_canopy_cover = (canopy_pixels / total_pixels) * 100
            
            # Calculate statistics
            canopy_covers = [img['canopy_cover'] for img in image_stats]
            min_cover = min(canopy_covers) if canopy_covers else 0
            max_cover = max(canopy_covers) if canopy_covers else 0
            mean_cover = np.mean(canopy_covers) if canopy_covers else 0
            std_cover = np.std(canopy_covers) if canopy_covers else 0
            
            city_stats[city] = {
                'overall_canopy_cover': overall_canopy_cover,
                'total_pixels': total_pixels,
                'canopy_pixels': canopy_pixels,
                'num_images': num_images,
                'min_cover': min_cover,
                'max_cover': max_cover,
                'mean_cover': mean_cover,
                'std_cover': std_cover,
                'image_stats': image_stats
            }
            
            print(f"  ✓ {city}: {overall_canopy_cover:.2f}% canopy cover "
                  f"({num_images} images, {total_pixels:,} total pixels)")
        else:
            print(f"  ✗ {city}: No valid images processed")
    
    # Print summary table
    print("\n" + "="*80)
    print("CANOPY COVER SUMMARY BY CITY")
    print("="*80)
    print(f"{'City':<15} {'Cover %':<12} {'Images':<10} {'Total Pixels':<15} {'Mean ± Std':<15}")
    print("-"*80)
    
    # Sort by canopy cover (highest first)
    sorted_stats = sorted(city_stats.items(), key=lambda x: x[1]['overall_canopy_cover'], reverse=True)
    
    for city, stats in sorted_stats:
        print(f"{city:<15} {stats['overall_canopy_cover']:>10.2f}%  {stats['num_images']:>8}  "
              f"{stats['total_pixels']:>14,}  {stats['mean_cover']:>6.2f} ± {stats['std_cover']:>5.2f}")
    
    print("="*80)
    
    # Print detailed breakdown
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY CITY")
    print("="*80)
    
    for city, stats in sorted_stats:
        print(f"\n{city}:")
        print(f"  Overall canopy cover: {stats['overall_canopy_cover']:.2f}%")
        print(f"  Based on {stats['num_images']} images")
        print(f"  Total pixels analyzed: {stats['total_pixels']:,}")
        print(f"  Canopy pixels: {stats['canopy_pixels']:,}")
        print(f"  Per-image statistics:")
        print(f"    - Mean: {stats['mean_cover']:.2f}%")
        print(f"    - Std: {stats['std_cover']:.2f}%")
        print(f"    - Min: {stats['min_cover']:.2f}%")
        print(f"    - Max: {stats['max_cover']:.2f}%")
    
    return city_stats

# Calculate canopy cover for each city (commented out for batch processing)
# Uncomment if you want to run this analysis
# canopy_stats = calculate_canopy_cover_by_city(cutouts_dir=CUTOUTS_DIR, threshold=0.5)
