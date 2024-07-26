import cv2
import numpy as np

def process_image(image_path, output_path):
  img = cv2.imread(image_path)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower = np.array([0, 20, 70], dtype="uint8")
  upper = np.array([20, 255, 255], dtype="uint8")
  mask = cv2.inRange(hsv, lower, upper)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Focus on the largest contour (assuming it's the hand or finger)
        largest_contour = contours[0]

        # Create a binary mask using the largest contour
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)

        # Perform post-processing to further improve the quality of the segmented region
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Segment the hand or finger region from the rest of the image
        hand = cv2.bitwise_and(img, img, mask=mask)

        # Convert to grayscale and apply adaptive thresholding
        hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        hand_thresh = cv2.adaptiveThreshold(hand_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image around the bounding box with some padding
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1], x + w + 2 * padding)
        h = min(img.shape[0], y + h + 2 * padding)
        cropped_hand = hand_thresh[y:h, x:w]

        # Save the cropped segmented region image
        cv2.imwrite(output_path, cropped_hand)
  else:
        print("No contours found!")