import cv2
import numpy as np

def preprocess_image(image):
    # Apply Median Filtering to each channel
    for i in range(3):
        image[..., i] = cv2.medianBlur(image[..., i], 5)
    
    # Optionally, apply Bilateral Filtering to each channel
    for i in range(3):
        image[..., i] = cv2.bilateralFilter(image[..., i], d=9, sigmaColor=75, sigmaSpace=75)

    return image