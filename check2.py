from rembg import remove
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

def remove_background(image_path):
    # Read image
    input_image = cv.imread(image_path)
    
    # Remove background
    output_image = remove(input_image)
    
    # Convert numpy array to PIL Image and back to numpy array
    output_image_pil = Image.fromarray(output_image)
    output_image_np = np.array(output_image_pil)

    return output_image_np

def zoom_in_on_contour(img, contour, padding=10):
    # Find the bounding box of the contour
    x, y, w, h = cv.boundingRect(contour)

    # Apply padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    # Crop the image around the bounding box
    cropped_img = img[y:y+h, x:x+w]

    return cropped_img

def process_image(image_path, output_path):
    # First, remove the background
    img = remove_background(image_path)
    
    # Convert to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define the lower and upper bounds for skin color in HSV
    lower = np.array([0, 20, 70], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # Threshold the image to extract only the skin color pixels
    mask = cv.inRange(hsv, lower, upper)

    # Perform morphological transformations to remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Detect contours in the binary image
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort the contours according to area
    if contours:
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # Focus on the largest contour (assuming it's the hand or finger)
        largest_contour = contours[0]

        # Zoom in on the largest contour
        zoomed_img = zoom_in_on_contour(img, largest_contour)

        # Convert the zoomed image to HSV color space
        hsv_zoomed = cv.cvtColor(zoomed_img, cv.COLOR_BGR2HSV)

        # Threshold the zoomed image to extract only the skin color pixels
        mask_zoomed = cv.inRange(hsv_zoomed, lower, upper)

        # Perform morphological transformations to remove noise
        mask_zoomed = cv.morphologyEx(mask_zoomed, cv.MORPH_CLOSE, kernel)

        # Detect contours in the binary image of the zoomed image
        contours_zoomed, _ = cv.findContours(mask_zoomed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Check the number of contours (fingers) detected
        if len(contours_zoomed) > 1:
            raise ValueError("More than one finger detected in the image!")

        # Create a binary mask using the largest contour
        if contours_zoomed:
            largest_contour_zoomed = sorted(contours_zoomed, key=cv.contourArea, reverse=True)[0]
            mask_final = np.zeros_like(mask_zoomed)
            cv.drawContours(mask_final, [largest_contour_zoomed], 0, 255, -1)

            # Perform post-processing to further improve the quality of the segmented region
            mask_final = cv.GaussianBlur(mask_final, (5, 5), 0)

            # Segment the hand or finger region from the rest of the image
            hand = cv.bitwise_and(zoomed_img, zoomed_img, mask=mask_final)

            # Convert to grayscale and apply adaptive thresholding
            hand_gray = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
            hand_thresh = cv.adaptiveThreshold(hand_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

            # Save the segmented region image
            cv.imwrite(output_path, hand_thresh)

            # Display the results
            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.title('Segmented Region')
            plt.imshow(hand_thresh, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Original Image with Mask')
            plt.imshow(cv.cvtColor(zoomed_img, cv.COLOR_BGR2RGB))
            plt.axis('off')

            plt.show()
        else:
            print("No contours found in the zoomed image!")
    else:
        print("No contours found!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = "images/output2.jpeg"
    process_image(image_path, output_path)
