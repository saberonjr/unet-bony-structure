
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Plot training history
def plot_training_history(history, results_folder):
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


    # Optionally save the training logs (history) to a text file
    history_save_path = os.path.join(results_folder, 'training_history.txt')
    with open(history_save_path, 'w') as f:
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")
    print(f"Training history saved to {history_save_path}")