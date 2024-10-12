
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def split_scoliosis_dataset(dataset, test_size=0.2, batch_size=8):
    # Convert `tf.data.Dataset` to lists of images and masks
    def dataset_to_numpy(dataset):
        images = []
        masks = []
        
        for batch_images, batch_masks in dataset:
            # Iterate through the images and masks in each batch
            for i in range(batch_images.shape[0]):  # Loop over the batch size
                images.append(batch_images[i].numpy())  # Convert Tensor to NumPy for each image
                masks.append(batch_masks[i].numpy())    # Convert Tensor to NumPy for each mask
        
        return np.array(images), np.array(masks)

    # Assuming `train_dataset` is a `tf.data.Dataset` object
    images, masks = dataset_to_numpy(dataset)

    # Split the dataset into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    # Convert back to `tf.data.Dataset`
    train_dataset_split = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_dataset_split = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

    # Batch and prefetch the datasets
    train_dataset_split = train_dataset_split.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset_split = val_dataset_split.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset_split, val_dataset_split