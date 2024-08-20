# import cv2
# import numpy as np

# def process_image(image_path, output_path):
#   img = cv2.imread(image_path)
#   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#   lower = np.array([0, 20, 70], dtype="uint8")
#   upper = np.array([20, 255, 255], dtype="uint8")
#   mask = cv2.inRange(hsv, lower, upper)

#   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   if contours:
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)

#         # Focus on the largest contour (assuming it's the hand or finger)
#         largest_contour = contours[0]

#         # Create a binary mask using the largest contour
#         mask = np.zeros_like(mask)
#         cv2.drawContours(mask, [largest_contour], 0, 255, -1)

#         # Perform post-processing to further improve the quality of the segmented region
#         mask = cv2.GaussianBlur(mask, (5, 5), 0)

#         # Segment the hand or finger region from the rest of the image
#         hand = cv2.bitwise_and(img, img, mask=mask)

#         # Convert to grayscale and apply adaptive thresholding
#         hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
#         hand_thresh = cv2.adaptiveThreshold(hand_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#         # Find the bounding box of the largest contour
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # Crop the image around the bounding box with some padding
#         padding = 30
#         x = max(0, x - padding)
#         y = max(0, y - padding)
#         w = min(img.shape[1], x + w + 2 * padding)
#         h = min(img.shape[0], y + h + 2 * padding)
#         cropped_hand = hand_thresh[y:h, x:w]

#         # Save the cropped segmented region image
#         cv2.imwrite(output_path, cropped_hand)
#   else:
#         print("No contours found!")



# import cv2
# import numpy as np
# from rembg import remove
# from PIL import Image
# from skimage import exposure

# def process_image(image_path, output_path):
#     # Read and remove background
#     input_image = cv2.imread(image_path)
#     output_image = remove(input_image)
    
#     # Convert numpy array to PIL Image and back to numpy array
#     output_image_pil = Image.fromarray(output_image)
#     output_image_np = np.array(output_image_pil)
    
#     # Convert to HSV
#     hsv = cv2.cvtColor(output_image_np, cv2.COLOR_BGR2HSV)
    
#     # Improve the quality of the image
#     # Enhance contrast
#     img_yuv = cv2.cvtColor(output_image_np, cv2.COLOR_BGR2YUV)
#     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#     img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
#     # Convert to HSV for segmentation
#     hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
#     lower = np.array([0, 20, 70], dtype="uint8")
#     upper = np.array([20, 255, 255], dtype="uint8")
#     mask = cv2.inRange(hsv, lower, upper)
    
#     # Morphological operations to clean up mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
#     # Find contours and sort them by area
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
#         # Focus on the largest contour
#         largest_contour = contours[0]
        
#         # Create a binary mask using the largest contour
#         mask = np.zeros_like(mask)
#         cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
#         # Perform post-processing
#         mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
#         # Segment the hand or finger region
#         hand = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)
        
#         # Convert to grayscale and apply adaptive thresholding
#         hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
#         hand_thresh = cv2.adaptiveThreshold(hand_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
#         # Find the bounding box of the largest contour
#         x, y, w, h = cv2.boundingRect(largest_contour)
        
#         # Crop the image around the bounding box with some padding
#         padding = 30
#         x = max(0, x - padding)
#         y = max(0, y - padding)
#         w = min(img_enhanced.shape[1], x + w + 2 * padding)
#         h = min(img_enhanced.shape[0], y + h + 2 * padding)
#         cropped_hand = hand_thresh[y:h, x:w]
        
#         # Save the cropped segmented region image
#         cv2.imwrite(output_path, cropped_hand)
#     else:
#         print("No contours found!")
import cv2
import numpy as np
# from rembg import remove
from PIL import Image, ImageFilter
from skimage import exposure, img_as_ubyte

def process_image(image_path, output_path):
    # Read and remove background
    input_image = cv2.imread(image_path)
    # output_image = remove(input_image)
    
    # Convert numpy array to PIL Image and back to numpy array
    # output_image_pil = Image.fromarray(output_image)
    # output_image_np = np.array(output_image_pil)

    # Enhance image quality
    img_yuv = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Additional enhancement (using CLAHE for better contrast)
    lab = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_enhanced = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    # Convert to HSV for segmentation
    hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and sort them by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Focus on the largest contour
        largest_contour = contours[0]
        
        # Create a binary mask using the largest contour
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Perform post-processing
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Segment the hand or finger region
        hand = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)
        
        # Convert to grayscale and apply adaptive thresholding
        hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        hand_thresh = cv2.adaptiveThreshold(hand_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply edge detection
        edges = cv2.Canny(hand_gray, 100, 200)
        
        # Find the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image around the bounding box with some padding
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_enhanced.shape[1], x + w + 2 * padding)
        h = min(img_enhanced.shape[0], y + h + 2 * padding)
        cropped_hand = hand_thresh[y:h, x:w]
        
        # Save the cropped segmented region image
        cv2.imwrite(output_path, cropped_hand)
    else:
        print("No contours found!")































