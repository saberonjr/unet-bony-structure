import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import segmentation_models as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For creating custom legends
from tensorflow.keras.utils import plot_model

def visualize_model(model, save_path):
    # Load the pre-trained model
   # model = keras.models.load_model(f'{save_path}/best_model.keras', compile=False)
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

    # Visualize and save the model architecture to a file
    plot_model(model, to_file=f'{save_path}/unet_model_visualization.png', show_shapes=True, show_layer_names=True)

    print("Model visualization saved to 'unet_model_visualization.png'")


if __name__ == '__main__':
    # Define the path to the saved model
    # ResNet backbone
    #save_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/BATCH8-3008hx640w-2024-10-11_10-34-23"
    
    # Base model
    save_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Model/BATCH8-320hx64w-2024-10-12_19-50-36-5Over100"

    # Load the pre-trained model
    model = keras.models.load_model(f'{save_path}/best_model.keras', compile=False)

    # Visualize the model
    visualize_model(model, save_path)