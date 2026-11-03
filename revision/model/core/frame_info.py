# Edited by Sizhuo Li
# Author: Ankit Kariryaa, University of Bremen

import numpy as np

def image_normalize(im, axis=(0, 1), c=1e-8):
    """Normalize to zero mean and unit standard deviation along the given axis.

    Args:
        im: ndarray
            Input image to normalize.
        axis: tuple of int
            Axes along which to compute the mean and std. Default is (0, 1).
        c: float
            Small constant to prevent division by zero. Default is 1e-8.

    Returns:
        ndarray: Normalized image.
    """
    return (im - im.mean(axis)) / (im.std(axis) + c)

class FrameInfo:
    """Defines a frame, including its constituent images, annotations, and weights (for weighted loss)."""

    def __init__(self, img, annotations, weight, dtype=np.float32):
        """FrameInfo constructor.

        Args:
            img: ndarray
                3D array containing various input channels.
            annotations: ndarray
                3D array containing human labels; height and width must be the same as img.
            weight: ndarray
                3D array containing weights for certain losses.
            dtype: np.float32, optional
                Datatype of the array (default: np.float32).
        """
        self.img = img
        self.annotations = annotations
        self.weight = weight
        self.dtype = dtype

    def getPatch(self, i, j, patch_size, img_size, normalize=1.0):
        """Function to get a patch from the given location of the specified size.

        Args:
            i: int
                Starting location on the first dimension (x axis).
            j: int
                Starting location on the second dimension (y axis).
            patch_size: tuple(int, int)
                Size of the patch to return.
            img_size: tuple(int, int)
                Size of the images from which the patch is generated.
            normalize: float
                Probability between 0 and 1 for normalizing the image.

        Returns:
            ndarray: Extracted patch containing the combined image, annotations, and weights.
        """
        patch = np.zeros(patch_size, dtype=self.dtype)
        im = self.img[i:i + img_size[0], j:j + img_size[1]]
        r = np.random.random(1)
        if normalize >= r[0]:
            im = image_normalize(im, axis=(0, 1))
        
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        an = np.expand_dims(an, axis=-1)
        we = self.weight[i:i + img_size[0], j:j + img_size[1]]
        we = np.expand_dims(we, axis=-1)
        
        comb_img = np.concatenate((im, an, we), axis=-1)
        patch[:img_size[0], :img_size[1]] = comb_img
        return patch

    def sequential_patches(self, patch_size, step_size, normalize):
        """Returns all sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Step size for moving along the image dimensions.
            normalize: float
                Probability with which a frame is normalized.

        Returns:
            list: List of extracted image patches.
        """
        img_shape = self.img.shape
        x = range(0, img_shape[0] - patch_size[0], step_size[0])
        y = range(0, img_shape[1] - patch_size[1], step_size[1])
        
        if img_shape[0] <= patch_size[0]:
            x = [0]
        if img_shape[1] <= patch_size[1]:
            y = [0]

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        
        img_patches = []
        for i, j in xy:
            img_patch = self.getPatch(i, j, patch_size, ic, normalize)
            img_patches.append(img_patch)
        
        return img_patches

    def random_patch(self, patch_size, normalize):
        """Returns a random patch from this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            normalize: float
                Probability with which a frame is normalized.

        Returns:
            ndarray: Randomly extracted image patch containing combined image, annotations, and weights.
        """
        img_shape = self.img.shape
        x = np.random.randint(0, img_shape[0] - patch_size[0]) if img_shape[0] > patch_size[0] else 0
        y = np.random.randint(0, img_shape[1] - patch_size[1]) if img_shape[1] > patch_size[1] else 0
        
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch = self.getPatch(x, y, patch_size, ic, normalize)
        return img_patch
