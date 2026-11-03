#    Author: Ankit Kariryaa, University of Bremen
#updates via curosr
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 50% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ], random_order=True)
    return seq

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames."""

    def __init__(self, input_image_channel, patch_size, frame_list, frames, annotation_channel=[2, 3], augmenter=None):
        """Datagenerator constructor."""
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter

    # Return all training and label images and weights, generated sequentially with the given step size
    def all_sequential_patches(self, step_size, normalize=1):
        """Generate all patches from all assigned frames sequentially."""
        patches = []
        for fn in self.frame_list:
            frame = self.frames[fn]
            ps = frame.sequential_patches(self.patch_size, step_size, normalize)
            patches.extend(ps)
        data = np.array(patches)
        img = data[..., self.input_image_channel]
        y = data[..., self.annotation_channel]

        # y would have two channels, i.e. annotations and weights.
        ann = y[..., [0]]
        weights = y[..., [1]]
        weights[weights >= 0.5] = 10
        weights[weights < 0.5] = 1
        ann_joint = np.concatenate((ann, weights), axis=-1)

        # Debug: Check shapes of generated patches
        #print(f"Generated sequential patch shape (img): {img.shape}")
        #print(f"Generated sequential patch shape (ann_joint): {ann_joint.shape}")

        return img, ann_joint

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE, normalize):
        """Generate patches from random location in randomly chosen frames."""
        patches = []
        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            patch = frame.random_patch(self.patch_size, normalize)
            patches.append(patch)
        data = np.array(patches)

        img = data[..., self.input_image_channel]
        ann_joint = data[..., self.annotation_channel]

        # Debug: Check shapes of generated patches
        #print(f"Generated random patch shape (img): {img.shape}")
        #print(f"Generated random patch shape (ann_joint): {ann_joint.shape}")

        return img, ann_joint

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def random_generator(self, BATCH_SIZE, normalize=1):
        """Generator for random patches, yields random patches from random location in randomly chosen frames."""
        seq = imageAugmentationWithIAA()

        while True:
            X, y = self.random_patch(BATCH_SIZE, normalize)

            # Debug: Check batch shapes before augmentation
            print(f"Batch shape before augmentation (X): {X.shape}, (y): {y.shape}")

            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
                # y would have two channels, i.e. annotations and weights. We need to augment y for operations such as crop and transform
                y = seq_det.augment_images(y)

                # Ensure labels are either 0 or 1
                ann = y[..., [0]]
                ann[ann < 0.5] = 0
                ann[ann >= 0.5] = 1

                # Boundaries have a weight of 10; other parts have a weight of 1
                weights = y[..., [1]]
                weights[weights >= 0.5] = 10
                weights[weights < 0.5] = 1

                ann_joint = np.concatenate((ann, weights), axis=-1)

                # Debug: Check batch shapes after augmentation
                #print(f"Batch shape after augmentation (X): {X.shape}, (ann_joint): {ann_joint.shape}")

                yield X, ann_joint
            else:
                # y would have two channels, i.e. annotations and weights.
                ann = y[..., [0]]
                weights = y[..., [1]]
                weights[weights >= 0.5] = 10
                weights[weights < 0.5] = 1

                ann_joint = np.concatenate((ann, weights), axis=-1)

                # Debug: Check batch shapes for non-augmented data
                #print(f"Batch shape without augmentation (X): {X.shape}, (ann_joint): {ann_joint.shape}")

                yield X, ann_joint
