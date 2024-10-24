import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import albumentations as A
from tensorflow.keras import layers, models
import segmentation_models as sm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Check if GPU is available and print details
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("TensorFlow is utilizing the GPU.")
    # Optional: Limit GPU memory growth
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
        
# Add the root directory (project_folder) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictions.predict import segment_and_save_results
from predictions.history import plot_training_history
from predictions.splitdataset import split_scoliosis_dataset

# Get only the file name without path and extension
current_file_name = os.path.basename(__file__)
sub_folder_name = os.path.splitext(current_file_name)[0]
print(f"Current file name without extension: {sub_folder_name}")

# Define paths
IMAGE_DIR = './Dataset/augmented_large/images/'
MASK_DIR = './Dataset/augmented_large/masks/'
#IMAGE_DIR = './Dataset/augmented_small/images/'
#MASK_DIR = './Dataset/augmented_small/masks/'
IMAGE_HEIGHT =   3008 # 320 #3008
IMAGE_WIDTH = 640 # 64 #640
BATCH_SIZE = 4 # #8 #32
TRAIN_LENGTH = len(os.listdir(IMAGE_DIR))
EPOCHS = 100

# Create a unique folder name based on the current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f'./Results/{sub_folder_name}/BATCH{BATCH_SIZE}-{IMAGE_HEIGHT}Hx{IMAGE_WIDTH}W-EPOCHS{EPOCHS}-{current_time}'
os.makedirs(results_folder, exist_ok=True)  # Create the folder if it doesn't exist


# Albumentations augmentations
augmentation_pipeline = A.Compose([
    #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
   #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.Rotate(limit=30, p=0.5),
    #A.ElasticTransform(p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)  # Ensures that the dimensions remain the same
])
"""
augmentation_pipeline = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        #A.Transpose(p=1),
    # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3), # careful with this one
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    ],is_check_shapes=False)
"""

# Function to load and augment image and mask
def load_data(image_path, mask_path, show_info=False):
    # Load image and mask
    image = Image.open(image_path).convert('RGB')  # 3 channels
    mask = Image.open(mask_path).convert('L')      # 1 channel

    # Convert to numpy arrays
    image = np.array(image)
    mask = np.array(mask)

    if show_info:
        # Print original dimensions of the image and mask
        print(f"Original Image Dimensions (H x W x C): {image.shape}")
        print(f"Original Mask Dimensions (H x W): {mask.shape}")


    # Apply augmentations
    augmented = augmentation_pipeline(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']

    if show_info:
        # Print original dimensions of the image and mask
        print(f"Augmented Image Dimensions (H x W x C): {image.shape}")
        print(f"Augmented Mask Dimensions (H x W): {mask.shape}")



    # Normalize image and mask
    image = image / 255.0  # Normalize image to [0, 1]
    mask = mask / 255.0    # Normalize mask to [0, 1]

    # Expand dimensions for masks to match the input shape
    mask = np.expand_dims(mask, axis=-1)

    return image, mask

# Data generator function
def data_generator(image_dir, mask_dir):
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace('image','mask').replace('.bmp', '.png'))
        yield load_data(image_path, mask_path)

# =====
# Test the data generator

# Function to test the data_generator and display the images and masks
def test_data_generator(image_dir, mask_dir, num_samples=3):
    # Instantiate the generator
    generator = data_generator(image_dir, mask_dir)
    
    # Loop through a few samples from the generator and display the images and masks
    print(f"Testing data_generator with {num_samples} samples...\n")
    for i in range(num_samples):
        try:
            image, mask = next(generator)  # Get the next image and mask from the generator
            print(f"Sample {i + 1}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Mask shape: {mask.shape}\n")

            # Plot the image and the mask side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Display the image (RGB)
            axs[0].imshow(image)
            axs[0].set_title('Image')
            axs[0].axis('off')

            # Display the mask (Grayscale)
            axs[1].imshow(mask[:, :, 0], cmap='gray')
            axs[1].set_title('Mask')
            axs[1].axis('off')

            plt.show()  # Display the plot
        except StopIteration:
            print("No more data available in the generator.")
            break


# Create tf.data.Dataset object
def create_dataset(image_dir, mask_dir, batch_size=8):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_dir, mask_dir),
        output_types=(tf.float32, tf.float32),
        output_shapes=((IMAGE_HEIGHT, IMAGE_WIDTH, 3), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    )
    dataset = dataset.batch(batch_size)
    return dataset

# Create train and validation datasets
train_dataset = create_dataset(IMAGE_DIR, MASK_DIR, batch_size=BATCH_SIZE)

train_dataset_split, val_dataset_split = split_scoliosis_dataset(train_dataset, test_size=0.2, batch_size=BATCH_SIZE)

print(train_dataset)


# =============================================================================
# # Define the U-Net model

def unet_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.Dropout(0.5)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.Dropout(0.5)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.Dropout(0.5)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.Dropout(0.5)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.Dropout(0.5)(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_model_batch(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_classes=1):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.3)(p1)  # Add dropout to prevent overfitting on noise

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.3)(p2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.3)(p3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.3)(p4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    outputs = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
# =============================================================================
#  Add Losses, Metrics, Checkpoints, and Early Stopping

# Define loss and metrics
dice_loss = sm.losses.DiceLoss()
jaccard_loss = sm.losses.JaccardLoss()
focal_loss = sm.losses.binary_focal_loss
#total_loss = dice_loss + jaccard_loss + focal_loss
total_loss = sm.losses.bce_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5,beta=2)]
metrics =[    
    sm.metrics.IOUScore(threshold=0.5),   # Intersection over Union (IoU)
    #sm.metrics.FScore(threshold=0.5,beta=2),     # F-Score (beta=1, F1-score)
    sm.metrics.f1_score,
    sm.metrics.f2_score,
    sm.metrics.precision,
    sm.metrics.recall
]
# Create U-Net model
model = unet_model_batch()

# Compile the model with the loss and metrics
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

# Define callbacks for ModelCheckpoint and EarlyStopping
checkpoint_path = os.path.join(results_folder, 'best_model.keras')
checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                             monitor='val_loss',  # Monitor validation loss
                             verbose=1,
                             save_best_only=True,  # Save only the best model
                             mode='min')  # 'min' because we want to minimize loss
# Callbacks for saving the best model and early stopping
# checkpoint_cb = ModelCheckpoint('./Results/Unet-Base/Today/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping_cb = EarlyStopping(patience=5, monitor='val_loss', verbose=1, restore_best_weights=True)


# Train the model with callbacks
history = model.fit(train_dataset_split,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,  # Increased epochs to allow early stopping
                    verbose=1,
                    validation_data=val_dataset_split,
                    callbacks=[checkpoint_cb, early_stopping_cb])  # Add callbacks here


# Save the model to the unique results folder
model_save_path = os.path.join(results_folder, 'scoliosis.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training history
plot_training_history(history, results_folder)

# Call the function with the save path
#segment_and_save_results(save_path = results_folder, target_width = IMAGE_WIDTH, target_height= IMAGE_HEIGHT,
#                         test_image= './Dataset/test/images/PWH00200114920160113006P5.bmp',
#                        test_mask='./Dataset/test/masks/PWH00200114920160113006P5.png',
#                        save_images=True)
segment_and_save_results(save_path = results_folder, target_width = IMAGE_WIDTH, target_height= IMAGE_HEIGHT,
                        test_image=f'./Dataset/test/images/PWH00200114920160113006P5.bmp',
                        test_mask=f'./Dataset/test/masks/PWH00200114920160113006P5.png',
                        save_images=True)
segment_and_save_results(save_path = results_folder, target_width = IMAGE_WIDTH, target_height= IMAGE_HEIGHT,
                        test_image=f'./Dataset/test/images/PWH00200114820160113005P5.bmp',
                        test_mask=f'./Dataset/test/masks/PWH00200114820160113005P5.png',
                        save_images=True)
segment_and_save_results(save_path = results_folder, target_width = IMAGE_WIDTH, target_height= IMAGE_HEIGHT,
                    test_image=f'./Dataset/test/images/PWH00200116320160115006P4.bmp',
                    test_mask=f'./Dataset/test/masks/PWH00200116320160115006P4.png',
                    save_images=True)
                   