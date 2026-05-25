import cv2
import numpy as np

def pencil_sketch(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image
    inverted = cv2.bitwise_not(gray)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Blend using color dodge
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    return sketch