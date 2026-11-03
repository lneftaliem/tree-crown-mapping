"""
unet_infer.py -- inference functions copied VERBATIM from cells 3, 4 and 6 of
$GROUP_SCRATCH/myenv_scratch/notebooks_scratch/3-RasterAnalysis.ipynb (the notebook that produced the
original predictions; extracted 2026-10-31). Only this header was added: the imports those cells rely
on (as in the notebook's first cell) and a module-level `model`, which the caller sets before calling detect_tree().
"""
import numpy as np
from itertools import product
from rasterio import windows
from tqdm import tqdm
import tensorflow as tf
from core.frame_info import image_normalize  # imported in the notebook's first cell (caller puts the notebook dir on sys.path)

model = None

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

