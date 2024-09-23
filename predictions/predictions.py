import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import segmentation_models as sm
import matplotlib.patches as mpatches  # For creating custom legends

def segment_and_save_results(save_path):
    # Adjust SIZE_X and SIZE_Y to match the input shape your U-Net model was trained on
    SIZE_X = 640   # Width that the model expects
    SIZE_Y = 3008  # Height that the model expects

    # Load the pre-trained model
    model = keras.models.load_model('/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/2024-09-23_09-05-08/best_model.keras', compile=False)

    # Compile the model with the same loss and metrics used during training
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

    # Recompile the loaded model with the loss and metrics
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

    # Test on a new image
    # Load and preprocess the test image
    test_img_path = '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test/images/PWH00200114920160113006P5.bmp'
    test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    original_image = cv2.resize(test_img, (SIZE_X, SIZE_Y))  # Resize to match model's input size

    test_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Load the original ground truth mask
    mask_path = '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test/masks/PWH00200114920160113006P5.png'
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    original_mask = cv2.resize(original_mask, (SIZE_X, SIZE_Y))

    # Expand dimensions to match model's input shape (1, height, width, channels)
    test_img_expanded = np.expand_dims(test_img, axis=0)
    original_mask_expanded = np.expand_dims(original_mask, axis=-1)
    original_mask_expanded = np.expand_dims(original_mask_expanded, axis=0)

    # Perform prediction
    prediction = model.predict(test_img_expanded)

    # Remove batch and channel dimensions from prediction
    predicted_mask = prediction[0, :, :, 0]

    # Threshold the predicted mask (if needed) to binary mask (e.g., 0.5 threshold for binary segmentation)
    predicted_mask_thresholded = (predicted_mask > 0.5).astype(np.uint8)

    # Convert both the original mask and predicted mask to 3-channel images for overlaying
    original_mask_rgb = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2RGB)
    predicted_mask_rgb = cv2.cvtColor(predicted_mask_thresholded * 255, cv2.COLOR_GRAY2RGB)

    # Define colors for masks (for visualization)
    # Ground Truth mask: Red (255, 0, 0), Predicted Mask: Green (0, 255, 0)
    colored_original_mask = np.zeros_like(original_mask_rgb)
    colored_predicted_mask = np.zeros_like(predicted_mask_rgb)

    colored_original_mask[:, :, 0] = original_mask  # Red channel for original mask
    colored_predicted_mask[:, :, 1] = predicted_mask_thresholded * 255  # Green channel for predicted mask

    # Overlay both masks with transparency onto the original image
    overlay_image = cv2.addWeighted(test_img, 0.7, colored_original_mask, 0.3, 0)  # Red overlay
    overlay_image = cv2.addWeighted(overlay_image, 1, colored_predicted_mask, 0.3, 0)  # Green overlay

    # Plot the original image, original mask, predicted mask, and the overlay
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

    # Original image with both original and predicted mask overlays
    axs[3].imshow(overlay_image)
    axs[3].set_title('Image with Both Masks Overlay')
    axs[3].axis('off')

    # Create a custom legend
    red_patch = mpatches.Patch(color='red', label='Original Mask (Ground Truth)')
    green_patch = mpatches.Patch(color='green', label='Predicted Mask')
    axs[3].legend(handles=[red_patch, green_patch], loc='upper right', fontsize='medium')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Optionally save the images
    plt.imsave('./Results/original_image.jpg', test_img)
    plt.imsave('./Results/original_mask.jpg', original_mask, cmap='gray')
    plt.imsave('./Results/predicted_mask.jpg', predicted_mask, cmap='gray')
    plt.imsave('./Results/overlay_image.jpg', overlay_image)

    # Evaluate the model on the test image and mask
    test_img_expanded = test_img_expanded.astype('float32')
    original_mask_expanded = original_mask_expanded.astype('float32')
    results = model.evaluate(test_img_expanded, original_mask_expanded, verbose=0)
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

    # Save the figure to a file
    plt.savefig('./Results/test_loss_and_metrics.png')  # You can specify the file format and path

    plt.show()


segment_and_save_results('/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/2024-09-23_09-05-08')