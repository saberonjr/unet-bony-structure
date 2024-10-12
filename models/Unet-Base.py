import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import datetime
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import sys
import segmentation_models as sm
# Add the root directory (project_folder) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictions.predict import segment_and_save_results

# Define the plain U-Net architecture
def unet(input_size=(640, 3008, 3)):
    inputs = layers.Input(input_size)

    # Encoder: Contracting Path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder: Expanding Path
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Create a folder for saving results
current_file_name = os.path.basename(__file__)
sub_folder_name = os.path.splitext(current_file_name)[0]
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f'./Results/{sub_folder_name}/{current_time}'
os.makedirs(results_folder, exist_ok=True)

# Image dimensions
SIZE_X = 320  # Height
SIZE_Y = 64   # Width

# Directories for images and masks
image_directory = '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_small/images'
mask_directory = '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_small/masks'

# Load the images and masks
image_paths = sorted(glob.glob(os.path.join(image_directory, "*.png")))
mask_paths = sorted(glob.glob(os.path.join(mask_directory, "*.png")))
assert len(image_paths) == len(mask_paths), "Mismatch between number of images and masks"

# Preprocessing
train_images = []
train_masks = []

for img_path, mask_path in zip(image_paths, mask_paths):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, 0)
    print(f"Original Image: {img.shape}, Mask: {mask.shape}")
    img = cv2.resize(img, (SIZE_Y, SIZE_X))  # Resize to match model input
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
    print(f"Resized Image: {img.shape}, Mask: {mask.shape}")
    
    train_images.append(img)
    train_masks.append(mask)

train_images = np.array(train_images)
train_masks = np.array(train_masks)

# Convert masks to binary (0 or 1)
train_masks = (train_masks > 0).astype(np.float32)

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# Expand dimensions for the masks
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

# Build the plain U-Net model
model = unet(input_size=(SIZE_X, SIZE_Y, 3))

# Define loss functions and metrics
dice_loss = sm.losses.DiceLoss()
jaccard_loss = sm.losses.JaccardLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + jaccard_loss + focal_loss

metrics = [
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5, beta=2)
]

# Compile the model
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

# Set up callbacks for checkpointing and early stopping
checkpoint_path = os.path.join(results_folder, 'best_model.keras')
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, batch_size=8, epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint, early_stopping])

# Save the trained model
model_save_path = os.path.join(results_folder, 'unet_model.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

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

# Call the plot function to visualize training history
plot_training_history(history)

# Optionally save the training logs (history) to a text file
history_save_path = os.path.join(results_folder, 'training_history.txt')
with open(history_save_path, 'w') as f:
    for key in history.history.keys():
        f.write(f"{key}: {history.history[key]}\n")
print(f"Training history saved to {history_save_path}")

# Call the segmentation function (assuming segment_and_save_results is defined elsewhere)
from predictions.predict import segment_and_save_results
segment_and_save_results(results_folder)