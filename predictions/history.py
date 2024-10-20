
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_history_from_file( file_path):
    history = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.split(': ')
            # Convert the string representation of the list to an actual list
            value_list = value.strip().strip('[]').split(', ')
            history[key] = [float(v) for v in value_list]
    return history


# Plot training history
def plot_training_history(history, results_folder):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))

    if 'history' in history.keys():
        history = history['history']

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Metrics (IoU and F-Score in this case)
    plt.subplot(1, 2, 2)
    if 'iou_score' in history:
        plt.plot(history['iou_score'], label='Training IoU')
        plt.plot(history['val_iou_score'], label='Validation IoU')
    if 'f1-score' in history:
        plt.plot(history['f1-score'], label='Training F-Score')
        plt.plot(history['val_f1-score'], label='Validation F-Score')
    if 'f2-score' in history:
        plt.plot(history['f2-score'], label='Training F-Score')
        plt.plot(history['val_f2-score'], label='Validation F-Score')
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
    history_save_path = os.path.join(results_folder, 'training_history2.txt')
    with open(history_save_path, 'w') as f:
        for key in history.keys():
            f.write(f"{key}: {history[key]}\n")
    print(f"Training history saved to {history_save_path}")


if __name__ == '__main__':
    # Path to the txt file
    file_path = '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/BATCH8-320hx64w-2024-10-12_21-25-12/training_history.txt'
    results_folder = '/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Results/Unet-Resnet34/BATCH8-320hx64w-2024-10-12_21-25-12'


    # Load the training history
    history_data = load_history_from_file(file_path)
    
    # Plot the training history
    plot_training_history(history_data, results_folder)
