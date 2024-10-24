

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import segmentation_models as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For creating custom legends

def segment_and_save_results(save_path, target_width, target_height, test_image, test_mask, save_images=True):
    # Adjust SIZE_X and SIZE_Y to match the input shape your U-Net model was trained on
    SIZE_X = target_width #640 # 64 #320 # 640   # Width that the model expects
    SIZE_Y = target_height #3008 # 360 #1280 # 3008  # Height that the model expects

    file_name = os.path.splitext(os.path.basename(test_image))[0]

    # Load the pre-trained model
    #model = keras.models.load_model('/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/2024-09-23_15-33-46/best_model.keras', compile=False)
    model = keras.models.load_model(f'{save_path}/best_model.keras', compile=False)
    # Compile the model with the same loss and metrics used during training
    dice_loss = sm.losses.DiceLoss()
    jaccard_loss = sm.losses.JaccardLoss()
    focal_loss = sm.losses.BinaryFocalLoss()

    # Custom loss: Combine Focal, Jaccard, and Dice losses
    total_loss = dice_loss + jaccard_loss + focal_loss

    # Metrics
    metrics = [
        sm.metrics.IOUScore(threshold=0.5),   # Intersection over Union (IoU)
        sm.metrics.FScore(threshold=0.5, beta=2)      # F-Score
    ]

    # Recompile the loaded model with the loss and metrics
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

    # Test on a new image
    # Load and preprocess the test image
    test_img_path = test_image # '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test/images/PWH00200114920160113006P5.bmp'
    test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    original_image = cv2.resize(test_img, (SIZE_X, SIZE_Y))  # Resize to match model's input size

    test_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Load the original ground truth mask
    mask_path =  test_mask # '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test/masks/PWH00200114920160113006P5.png'
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

    # Save the images to the provided directory
    if save_images:
        plt.imsave(os.path.join(save_path, f'{file_name}-original_image.jpg'), test_img)
        plt.imsave(os.path.join(save_path, f'{file_name}-original_mask.jpg'), original_mask, cmap='gray')
        plt.imsave(os.path.join(save_path, f'{file_name}-predicted_mask.jpg'), predicted_mask, cmap='gray')
        plt.imsave(os.path.join(save_path, f'{file_name}-overlay_image.jpg'), overlay_image)

    # Evaluate the model on the test image and mask
    test_img_expanded = test_img_expanded.astype('float32')
    original_mask_expanded = original_mask_expanded.astype('float32')
    results = model.evaluate(test_img_expanded, original_mask_expanded, verbose=0)
    test_loss = results[0]  # Loss on the test data
    test_metrics = results[1:]  # Metrics on the test data

    # Assuming the model was compiled with IoU and F-Score as metrics
    metrics_names = model.metrics_names[1:]  # Exclude 'loss' from metrics names

    # Plot the loss
    plt.figure(figsize=(6, 6))
    plt.bar(['Test Loss'], [test_loss])
    plt.title('Test Loss')
    plt.tight_layout()

    # Save the loss figure to a file in the provided directory
    if save_images:
        plt.savefig(os.path.join(save_path, f'{file_name}-test_loss.png'))  # Save loss plot
    plt.show()

    # Separate IoU and F-Score and plot them individually
    for metric_name, metric_value in zip(metrics_names, test_metrics):
        plt.figure(figsize=(6, 6))
        plt.bar([metric_name], [metric_value])
        plt.title(f'Test {metric_name}')
        plt.tight_layout()

        # Save each metric figure individually
        if save_images:
            plt.savefig(os.path.join(save_path, f'{file_name}-test_{metric_name.lower()}.png'))  # Save each metric plot
        plt.show()
        

if __name__ == '__main__':
    target_width = 64
    target_height = 320
    model_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/2024-09-23_15-33-46"
    test_image_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test"
    """
    segment_and_save_results(model_path,
                             target_width, target_height, 
                             f'{test_image_path}/images/PWH00200114920160113006P5.bmp',
                             f'{test_image_path}/masks/PWH00200114920160113006P5.png',
                        True)
    """
    '''
    segment_and_save_results(model_path,
                             target_width, target_height, 
                             f'{test_image_path}/images/PWH00200114820160113005P5.bmp',
                             f'{test_image_path}/masks/PWH00200114820160113005P5.png',
                        True)
    segment_and_save_results('/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/2024-10-11_10-34-23c',
                             640, 3008, 
                             f'{test_image_path}/images/PWH00200116320160115006P4.bmp',
                             f'{test_image_path}/masks/PWH00200116320160115006P4.png',
                        True)
    '''

    # Unet-Model 320x64
    '''
    target_width = 64
    target_height = 320
    model_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Model/BATCH8-320hx64w-2024-10-12_19-50-36-5Over100"
    test_image_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test"
    segment_and_save_results(model_path,
                             target_width, target_height, 
                             f'{test_image_path}/images/PWH00200114920160113006P5.bmp',
                             f'{test_image_path}/masks/PWH00200114920160113006P5.png',
                        True)
    segment_and_save_results(model_path,
                              target_width, target_height,
                             f'{test_image_path}/images/PWH00200114820160113005P5.bmp',
                             f'{test_image_path}/masks/PWH00200114820160113005P5.png',
                        True)
    segment_and_save_results(model_path,
                              target_width, target_height,
                             f'{test_image_path}/images/PWH00200116320160115006P4.bmp',
                             f'{test_image_path}/masks/PWH00200116320160115006P4.png',
                        True)
    
    
    # Resnet34-Model 320x64
    target_width = 64
    target_height = 320
    model_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Model/BATCH8-320HX640W-2024-10-13_00-37-01-FINAL"
    test_image_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test"
    segment_and_save_results(model_path,
                             target_width, target_height, 
                             f'{test_image_path}/images/PWH00200114920160113006P5.bmp',
                             f'{test_image_path}/masks/PWH00200114920160113006P5.png',
                        True)
    segment_and_save_results(model_path,
                              target_width, target_height,
                             f'{test_image_path}/images/PWH00200114820160113005P5.bmp',
                             f'{test_image_path}/masks/PWH00200114820160113005P5.png',
                        True)
    segment_and_save_results(model_path,
                              target_width, target_height,
                             f'{test_image_path}/images/PWH00200116320160115006P4.bmp',
                             f'{test_image_path}/masks/PWH00200116320160115006P4.png',
                        True)
    
    # Resnet34-Model 320x64
    '''
    target_width = 640
    target_height = 3008
    model_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/BATCH4-3008Hx640W-EPOCHS100-2024-10-23_00-29-59"
    test_image_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/test"
    segment_and_save_results(model_path,
                             target_width, target_height, 
                             f'{test_image_path}/images/PWH00200114920160113006P5.bmp',
                             f'{test_image_path}/masks/PWH00200114920160113006P5.png',
                        True)
    segment_and_save_results(model_path,
                              target_width, target_height,
                             f'{test_image_path}/images/PWH00200114820160113005P5.bmp',
                             f'{test_image_path}/masks/PWH00200114820160113005P5.png',
                        True)
    segment_and_save_results(model_path,
                              target_width, target_height,
                             f'{test_image_path}/images/PWH00200116320160115006P4.bmp',
                             f'{test_image_path}/masks/PWH00200116320160115006P4.png',
                        True)
     