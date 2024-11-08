import random
import os
import albumentations as A
import cv2    

images_to_generate=1000

def display_images_masks(images_path, masks_path, images_to_generate):
    images = []
    masks = []

    for im in os.listdir(images_path):  # read image name from images path and append its path into "images" array     
        images.append(os.path.join(images_path,im))

    for msk in os.listdir(masks_path):  # read image name from masks path and append its path into "masks" array
        masks.append(os.path.join(masks_path,msk))


    i = 1
    height_max = 0
    height_min = 3500
    while i <= images_to_generate:
        number = random.randint(0, len(images)-1)  # Pick a number to select an image/mask
        image = images[number]
        mask = masks[number]
        #print(image, mask)


        original_image = cv2.imread(image)
        original_mask = cv2.imread(mask)

        image_height, image_width = original_image.shape[:2]  # [:2] gets (height, width) without the channel
        if (image_height > height_max):
            height_max = image_height
        if (image_height < height_min):
            height_min = image_height

        # Get the height and width of the mask (if it's grayscale, it's 2D)
        mask_height, mask_width = original_mask.shape[:2]

        print(f"Image dimensions: {image_height} x {image_width} Mask dimensions: {mask_height} x {mask_width}")

        i = i + 1

    print("Max Height: ", height_max, "Min Height: ", height_min)

    
if __name__ == "__main__":
    images_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/images/"
    masks_path = "/Users/soterojrsaberon/SeriousAI/BonyStructureSegmentation/Dataset/masks/"
    display_images_masks(images_path, masks_path, images_to_generate)
