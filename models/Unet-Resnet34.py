import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import datetime
import segmentation_models as sm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from segmentation_utils import segment_and_save_results


# Get only the file name without path and extension
current_file_name = os.path.basename(__file__)
sub_folder_name = os.path.splitext(current_file_name)[0]

print(f"Current file name without extension: {sub_folder_name}")


# Create a unique folder name based on the current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f'./Results/{sub_folder_name}/{current_time}'
os.makedirs(results_folder, exist_ok=True)  # Create the folder if it doesn't exist


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


#Resizing images is optional, CNNs are ok with large images
SIZE_X = 3008 #Resize images (height  = X, width = Y)
SIZE_Y = 640

image_directory = '../Dataset/augmented_new/images/'
mask_directory = '../Dataset/augmented_new/masks/'

# Capture image and mask info
image_paths = sorted(glob.glob(os.path.join(image_directory, "*.png")))  # Sort to maintain order
mask_paths = sorted(glob.glob(os.path.join(mask_directory, "*.png")))    # Sort to maintain order

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

# Custom loss: Combine Focal, Jaccard, and Dice losses
total_loss = dice_loss + jaccard_loss + focal_loss

# Metrics
metrics = [
    sm.metrics.IOUScore(threshold=0.5),   # Intersection over Union (IoU)
    sm.metrics.FScore(threshold=0.5)      # F-Score
]

# Define and compile U-Net model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
####

# define model
#model = sm.Unet(BACKBONE, encoder_weights='imagenet', )
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

print(model.summary())

"""
history=model.fit(x_train, 
          y_train,
          batch_size=4, 
          epochs=1, #5
          verbose=1,
          validation_data=(x_val, y_val))

"""
# Define callbacks for ModelCheckpoint and EarlyStopping
checkpoint_path = os.path.join(results_folder, 'best_model.keras')

checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             monitor='val_loss',  # Monitor validation loss
                             verbose=1,
                             save_best_only=True,  # Save only the best model
                             mode='min')  # 'min' because we want to minimize loss

early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=10,  # Stop after 10 epochs of no improvement
                               verbose=1,
                               restore_best_weights=True)  # Restore the best weights at the end

# Train the model with callbacks
history = model.fit(x_train,
                    y_train,
                    batch_size=8,
                    epochs=1,  # Increased epochs to allow early stopping
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, early_stopping])  # Add callbacks here

# Save the model to the unique results folder
model_save_path = os.path.join(results_folder, 'scoliosis.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


#model.save('scoliosis.keras') # creates a HDF5 file 'my_model.h5'

"""
#accuracy = model.evaluate(x_val, y_val)
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('scoliosis.keras') # creates a HDF5 file 'my_model.h5'
"""
# Plot training history
def plot_training_history(history):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Metrics (IoU and F-Score in this case)
    plt.subplot(1, 2, 2)
    if 'iou_score' in history.history:
        plt.plot(history.history['iou_score'], label='Training IoU')
        plt.plot(history.history['val_iou_score'], label='Validation IoU')
    if 'f1-score' in history.history:
        plt.plot(history.history['f1-score'], label='Training F-Score')
        plt.plot(history.history['val_f1-score'], label='Validation F-Score')
    plt.title('Metrics (IoU / F-Score)')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.tight_layout()
    # Save the plot to the results folder
    plot_save_path = os.path.join(results_folder, 'training_history.png')
    plt.savefig(plot_save_path)
    plt.show()
    print(f"Training history plot saved to {plot_save_path}")

# Assuming you have the history object after training your model
plot_training_history(history)

# Optionally save the training logs (history) to a text file
history_save_path = os.path.join(results_folder, 'training_history.txt')
with open(history_save_path, 'w') as f:
    for key in history.history.keys():
        f.write(f"{key}: {history.history[key]}\n")
print(f"Training history saved to {history_save_path}")


# Call the function with the save path
segment_and_save_results(results_folder)