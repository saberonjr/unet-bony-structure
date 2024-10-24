import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import datetime
import segmentation_models as sm
import glob
import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os

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
IMAGE_HEIGHT = 3008 # 320 #3008
IMAGE_WIDTH =  640 # 64 #640
BATCH_SIZE = 4 # 32 #8  
TRAIN_LENGTH = len(os.listdir(IMAGE_DIR))
EPOCHS =  100

# Create a unique folder name based on the current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f'./Results/{sub_folder_name}/BATCH{BATCH_SIZE}-{IMAGE_HEIGHT}Hx{IMAGE_WIDTH}W-EPOCHS{EPOCHS}-{current_time}'
os.makedirs(results_folder, exist_ok=True)  # Create the folder if it doesn't exist



BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

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

# Capture image and mask info
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))  # Sort to maintain order
mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))    # Sort to maintain order

# Ensure the number of images matches the number of masks
assert len(image_paths) == len(mask_paths), "Mismatch between number of images and masks"


#Capture training image info as a list
train_images = []
train_masks = [] 

# Loop through both the images and masks, ensuring they are paired by filename
for img_path, mask_path in zip(image_paths, mask_paths):
    # Extract the filenames without the directories
    img_filename = os.path.basename(img_path)
    mask_filename = os.path.basename(mask_path)

    # Ensure that the filenames match (if they have different prefixes, adjust accordingly)
    #assert img_filename == mask_filename, f"Image {img_filename} and Mask {mask_filename} do not match"

    # Read the image and mask
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load image in color
    mask = cv2.imread(mask_path, 0)  # Load mask as grayscale

    # Optionally resize the image and mask if required
    # img = cv2.resize(img, (SIZE_Y, SIZE_X))
    # mask = cv2.resize(mask, (SIZE_Y, SIZE_X))

    # Append to the respective lists
    train_images.append(img)
    train_masks.append(mask)


train_images = np.array(train_images)
train_masks = np.array(train_masks)

# Convert masks to binary (0 or 1)
train_masks = (train_masks > 0).astype(np.float32)

#Use customary x_train and y_train variables
X = train_images
Y = train_masks
#Y = np.expand_dims(Y, axis=3) #May not be necessary.


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

#######
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

# Loss functions and metrics
dice_loss = sm.losses.DiceLoss()
jaccard_loss = sm.losses.JaccardLoss()
focal_loss = sm.losses.BinaryFocalLoss()
#bce_loss = sm.losses.binary_crossentropy

# Custom loss: Combine Focal, Jaccard, and Dice losses
#total_loss = dice_loss + jaccard_loss + focal_loss
total_loss = sm.losses.bce_dice_loss

#total_loss = sm.binary_focal_dice_loss()
#total_loss = sm.binary_focal_jaccard_loss()


# Metrics (including Dice Score, IoU, and F-Score)
metrics = [
    sm.metrics.IOUScore(threshold=0.5),   # Intersection over Union (IoU)
    #sm.metrics.FScore(threshold=0.5,beta=2),     # F-Score (beta=1, F1-score)
    sm.metrics.f1_score,
    sm.metrics.f2_score,
    sm.metrics.precision,
    sm.metrics.recall, 
    #sm.metrics.accuracy
]



# Define and compile U-Net model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

print(model.summary())

# Define callbacks for ModelCheckpoint and EarlyStopping
checkpoint_path = os.path.join(results_folder, 'best_model.keras')

checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             monitor='val_loss',  # Monitor validation loss
                             verbose=1,
                             save_best_only=True,  # Save only the best model
                             mode='min')  # 'min' because we want to minimize loss

early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=5,  # Stop after 10 epochs of no improvement
                               verbose=1,
                               restore_best_weights=True)  # Restore the best weights at the end

# Train the model with callbacks
history = model.fit(x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,  # Increased epochs to allow early stopping
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, early_stopping])  # Add callbacks here

# Save the model to the unique results folder
model_save_path = os.path.join(results_folder, 'scoliosis.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training history
plot_training_history(history, results_folder)

# Call the function with the save path
#segment_and_save_results(results_folder, IMAGE_WIDTH, IMAGE_HEIGHT,
#                        '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test/images/PWH00200114920160113006P5.bmp',
#                        '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test/masks/PWH00200114920160113006P5.png',
#                        save_images=True)

# Call the function with the save path
'''
segment_and_save_results(save_path = results_folder, target_width = IMAGE_WIDTH, target_height= IMAGE_HEIGHT,
                         test_image= './Dataset/test/images/PWH00200114920160113006P5.bmp',
                        test_mask='./Dataset/test/masks/PWH00200114920160113006P5.png',
                        save_images=True)
'''
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