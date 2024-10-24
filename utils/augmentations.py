import random
import os
import albumentations as A
import cv2    


images_to_generate=1000

def augment_images(images_path, masks_path, image_augmented_path, mask_augmented_path, images_to_generate, target_height, target_width):    
                            

    #images_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/images/"
    #masks_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/masks/"
    #image_augmented_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_medium/images/"
    #mask_augmented_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_medium/masks/"

    images = []
    masks = []

    # Read images and masks, ensuring they are paired correctly
    image_filenames = sorted(os.listdir(images_path))  # Sort filenames to keep order
    mask_filenames = sorted(os.listdir(masks_path))    # Sort filenames to keep order

    # Ensure the number of images and masks match
    #assert len(image_filenames) == len(mask_filenames), "Mismatch between number of images and masks"

    # Loop through image and mask filenames and append the paths to lists
    for img_filename, mask_filename in zip(image_filenames, mask_filenames):
        # Ensure that image and mask filenames correspond (without extensions if necessary)
        img_base, img_ext = os.path.splitext(img_filename)
        mask_base, mask_ext = os.path.splitext(mask_filename)

        # Optional: If masks have a different prefix, adjust accordingly
        # For example, if the images are prefixed with "image_" and masks with "mask_"
        # img_base = img_base.replace("image_", "")
        # mask_base = mask_base.replace("mask_", "")

        #assert img_base == mask_base, f"Image {img_base} and mask {mask_base} do not match"

        # Append the full paths of matched image and mask
        images.append(os.path.join(images_path, img_filename))
        masks.append(os.path.join(masks_path, mask_filename))


    print(f"Number of images: {len(images)}")
    print(f"Number of masks: {len(masks)}")


    #target_height = 1280 # 3008
    #target_width = 320 # 640

    aug = A.Compose([
        A.Resize(height=target_height, width=target_width, p=1),
        #A.VerticalFlip(p=0.5),       ###       
        #A.RandomRotate90(p=0.5), ###
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        #A.Transpose(p=1),   ###
        A.ElasticTransform(alpha=0.5, sigma=40, alpha_affine=30, p=0.3), # careful with this one ###
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.3)
    ],is_check_shapes=False)

    i = 1

    while i <= images_to_generate:
        number = random.randint(0, len(images)-1)  # Pick a number to select an image/mask
        image = images[number]
        mask = masks[number]
        print(image, mask)


        original_image = cv2.imread(image)
        original_mask = cv2.imread(mask)

        image_height, image_width = original_image.shape[:2]  # [:2] gets (height, width) without the channel

        # Get the height and width of the mask (if it's grayscale, it's 2D)
        mask_height, mask_width = original_mask.shape[:2]

        print(f"Image dimensions: {image_height} x {image_width}")
        print(f"Mask dimensions: {mask_height} x {mask_width}")

        augmented = aug(image=original_image, mask=original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']

        image_height, image_width = transformed_image.shape[:2]  # [:2] gets (height, width) without the channel

        # Get the height and width of the mask (if it's grayscale, it's 2D)
        mask_height, mask_width = transformed_mask.shape[:2]

        print(f"Augmented Image dimensions: {image_height} x {image_width}")
        print(f"Augmented Mask dimensions: {mask_height} x {mask_width}")

        # Construct new file paths for the transformed images and masks
        new_image_path = os.path.join(image_augmented_path, f"augmented_image_{i}.png")
        new_mask_path = os.path.join(mask_augmented_path, f"augmented_mask_{i}.png")


        # Save the transformed image and mask to the file system
        cv2.imwrite(new_image_path, transformed_image)  # Save transformed image
        cv2.imwrite(new_mask_path, transformed_mask)    # Save transformed mask

        print(f"Saved augmented image to: {new_image_path}")
        print(f"Saved augmented mask to: {new_mask_path}")

        i = i + 1
    
if __name__ == "__main__":
    """
    images_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/images/"
    masks_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/masks/"
    image_augmented_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_large/images/"
    mask_augmented_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_large/masks/"
    augment_images(images_path, masks_path, image_augmented_path, mask_augmented_path, images_to_generate, 3008, 640)
    """
    images_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/images/"
    masks_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/masks/"
    image_augmented_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_small/images/"
    mask_augmented_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/augmented_small/masks/"
    augment_images(images_path, masks_path, image_augmented_path, mask_augmented_path, images_to_generate, 320, 64)
