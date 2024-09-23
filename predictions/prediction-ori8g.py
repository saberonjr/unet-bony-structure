from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Adjust SIZE_X and SIZE_Y to match the input shape your U-Net model was trained on
SIZE_X = 640   # Width that the model expects
SIZE_Y = 3008  # Height that the model expects

# Load the pre-trained model
model = keras.models.load_model('./scoliosis.h5', compile=False)

# Test on a new image
# Load and preprocess the test image
test_img_path = './Dataset/augmented2/images/augmented_image_1.png'
test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
original_image = cv2.resize(test_img, (SIZE_Y, SIZE_X))  # Resize to match model's input size
test_img = original_image # cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Load the original ground truth mask
mask_path = 'Dataset/augmented2/masks/augmented_mask_1.png'
original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
original_mask = cv2.resize(original_mask, (SIZE_Y, SIZE_X))

# Expand dimensions to match model's input shape (1, height, width, channels)
test_img_expanded = np.expand_dims(test_img, axis=0)
original_mask_expanded = np.expand_dims(original_mask, axis=-1)
original_mask_expanded = np.expand_dims(original_mask_expanded, axis=0)


# Expand dimensions to match model's input shape (1, height, width, channels)
test_img_expanded = np.expand_dims(test_img, axis=0)

# Perform prediction
prediction = model.predict(test_img_expanded)

# Remove batch and channel dimensions from prediction
predicted_mask = prediction[0, :, :, 0]

# Threshold the predicted mask (if needed) to binary mask (e.g., 0.5 threshold for binary segmentation)
predicted_mask_thresholded = (predicted_mask > 0.5).astype(np.uint8)

# Convert predicted mask to 3 channels to overlay with the original image
predicted_mask_rgb = cv2.cvtColor(predicted_mask_thresholded * 255, cv2.COLOR_GRAY2RGB)

# Overlay the predicted mask on the original image (with some transparency)
overlay_image = cv2.addWeighted(test_img, 0.7, predicted_mask_rgb, 0.3, 0)

# Plot the original image, predicted mask, original mask, and the overlay
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Original image
axs[0].imshow(test_img)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Original mask (ground truth)
axs[1].imshow(original_mask, cmap='gray')
axs[1].set_title('Original Mask (Ground Truth)')
axs[1].axis('off')

# Predicted mask
axs[2].imshow(predicted_mask, cmap='gray')
axs[2].set_title('Predicted Mask')
axs[2].axis('off')

# Original image with predicted mask overlay
axs[3].imshow(overlay_image)
axs[3].set_title('Image with Predicted Mask Overlay')
axs[3].axis('off')

# Show the plots
plt.tight_layout()
plt.show()

# Optionally save the images
plt.imsave('./Results/original_image.jpg', test_img)
plt.imsave('./Results/original_mask.jpg', original_mask, cmap='gray')
plt.imsave('./Results/predicted_mask.jpg', predicted_mask, cmap='gray')
plt.imsave('./Results/overlay_image.jpg', overlay_image)

"""
# Evaluate the model on the test image and mask
#results = model.evaluate(test_img_expanded, original_mask_expanded, verbose=0)
test_loss = results[0]  # Loss on the test data
test_metrics = results[1:]  # Metrics on the test data

# Assuming the model was compiled with IoU and F-Score as metrics
metrics_names = model.metrics_names[1:]  # Exclude 'loss' from metrics names

# Plot loss and metrics for the test data
plt.figure(figsize=(12, 6))

# Plot the loss
plt.subplot(1, 2, 1)
plt.bar(['Test Loss'], [test_loss])
plt.title('Test Loss')

# Plot the metrics (IoU, F-Score)
plt.subplot(1, 2, 2)
plt.bar(metrics_names, test_metrics)
plt.title('Test Metrics (IoU & F-Score)')

plt.tight_layout()
plt.show()
"""