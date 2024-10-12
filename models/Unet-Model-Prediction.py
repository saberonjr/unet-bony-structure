import matplotlib.pyplot as plt
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from segmentation_models import losses
from segmentation_models.losses import DiceLoss, JaccardLoss, binary_focal_loss
from segmentation_models.metrics import IOUScore, FScore
import numpy as np  
import albumentations as A
from PIL import Image

IMAGE_DIR = './Dataset/augmented_small/images/'
MASK_DIR = './Dataset/augmented_small/masks/'
IMAGE_HEIGHT = 320 #3008
IMAGE_WIDTH =64 #640
BATCH_SIZE = 8  
TRAIN_LENGTH = len(os.listdir(IMAGE_DIR))

# =============================================================================
# Test the Model on a New Image

# Define test paths
TEST_IMAGE_DIR = './Dataset/test/images/'
TEST_MASK_DIR = './Dataset/test/masks/'

augmentation_pipeline = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)  # Ensures that the dimensions are always 3008x640
])

# Function to load test images without augmentation
def load_test_data(image_path, mask_path):
    # Load image and mask
    image = Image.open(image_path).convert('RGB')  # 3 channels
    mask = Image.open(mask_path).convert('L')      # 1 channel

    # Convert to numpy arrays
    image = np.array(image)
    mask = np.array(mask)

    # Apply augmentations
    augmented = augmentation_pipeline(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']

    # Normalize image and mask
    image = image / 255.0  # Normalize image to [0, 1]
    mask = mask / 255.0    # Normalize mask to [0, 1]

    # Expand dimensions for masks to match the input shape
    mask = np.expand_dims(mask, axis=-1)

    return image, mask

# Data generator function for test data
def test_data_generator(image_dir, mask_dir):
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace('.bmp', '.png'))
        yield load_test_data(image_path, mask_path)

# Create tf.data.Dataset object for test data
def create_test_dataset(image_dir, mask_dir, batch_size=1):
    dataset = tf.data.Dataset.from_generator(
        lambda: test_data_generator(image_dir, mask_dir),
        output_types=(tf.float32, tf.float32),
        output_shapes=((IMAGE_HEIGHT, IMAGE_WIDTH, 3), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    )
    dataset = dataset.batch(batch_size)
    return dataset

# Create test dataset
test_dataset = create_test_dataset(TEST_IMAGE_DIR, TEST_MASK_DIR, batch_size=1)

# =============================================================================
# Predict on the Test Dataset


# Path to the saved model
model_path = './Results/Unet-Base/Today/best_model.keras'




# Define the custom SumOfLosses class
class SumOfLosses:
    def __init__(self, losses):
        self.losses = losses
    
    def __call__(self, y_true, y_pred):
        return sum([loss(y_true, y_pred) for loss in self.losses])
    
    def get_config(self):
        return {'losses': [loss.__class__.__name__ for loss in self.losses]}
    
    @classmethod
    def from_config(cls, config):
        loss_mapping = {
            'DiceLoss': DiceLoss(),
            'JaccardLoss': JaccardLoss(),
            'BinaryFocalLoss': binary_focal_loss
        }
        losses = [loss_mapping[loss_name] for loss_name in config['losses']]
        return cls(losses)

# Instantiate the SumOfLosses with the individual loss functions
dice_loss = DiceLoss()
jaccard_loss = JaccardLoss()
focal_loss = binary_focal_loss
sum_of_losses = SumOfLosses([dice_loss, jaccard_loss, focal_loss])


# Load the model with custom objects
model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        'SumOfLosses': sum_of_losses,  # Recreate SumOfLosses
        'dice_loss': dice_loss,
        'jaccard_loss': jaccard_loss,
        'binary_focal_loss': focal_loss,
        'IOUScore': IOUScore,
        'FScore': FScore
    }
)


# Load the model
model = tf.keras.models.load_model(model_path)

# Print the model summary to verify it has been loaded correctly
model.summary()

# Function to display and save results
def save_and_display_predictions(model, test_dataset, save_dir='./Results/Unet-Base/Today'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (image, mask) in enumerate(test_dataset):
        # Predict the mask using the trained model
        predicted_mask = model.predict(image)
        
        # Threshold the predicted mask to binary (assuming sigmoid output)
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        # Convert tensor to numpy for plotting
        image_np = image[0].numpy()
        mask_np = mask[0].numpy()
        predicted_mask_np = predicted_mask[0]

        # Overlay of original mask and predicted mask
        overlay = np.stack([predicted_mask_np[:, :, 0], mask_np[:, :, 0], np.zeros_like(mask_np[:, :, 0])], axis=-1)

        # Plot and save results
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(image_np)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(mask_np[:, :, 0], cmap='gray')
        axs[1].set_title('Original Mask')
        axs[1].axis('off')

        axs[2].imshow(predicted_mask_np[:, :, 0], cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        axs[3].imshow(overlay)
        axs[3].set_title('Overlay (Predicted vs Original)')
        axs[3].axis('off')

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"result_{i}.png"))
        plt.show()

# Run predictions on test dataset and save results
save_and_display_predictions(model, test_dataset, save_dir='./Results/Unet-Base/Today/')