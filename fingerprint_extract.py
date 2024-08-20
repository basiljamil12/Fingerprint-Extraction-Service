#Service::

from src.utils.fingers_extract import process_image
import os
import cv2
import numpy as np
import uuid
from datetime import datetime
from io import BytesIO

def process_fingerprint(file_stream, user_id):
    # Generate unique file names using timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex
    temp_input_path = f'temp_input_{timestamp}_{unique_id}.png'
    temp_output_path = f'temp_output_{timestamp}_{unique_id}.png'
    
    try:
        # Save the uploaded file to disk
        with open(temp_input_path, "wb") as f:
            f.write(file_stream.read())
        
        # Process the image
        process_image(temp_input_path, temp_output_path)
        
        # Read the processed image
        processed_image = cv2.imread(temp_output_path, cv2.IMREAD_GRAYSCALE)
        processed_image_bytes = cv2.imencode('.png', processed_image)[1].tobytes()
        
        # Prepare data to be returned
        data = {
            'user_id': user_id,
            'processed_image': processed_image_bytes.hex()  # convert to hex string
        }
        
        return data
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)



#Controller

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.services.fingerprint_extract_service import process_fingerprint
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

router = APIRouter()

class FingerprintRequest(BaseModel):
    userId: str
    finger: str  # URL of the image

@router.post("/extract")
async def extract_fingerprints(request: FingerprintRequest):
    user_id = request.userId
    image_url = request.finger
    
    if not user_id:
        raise HTTPException(status_code=400, detail="No userId provided in JSON data")
    
    if not image_url:
        raise HTTPException(status_code=400, detail="No finger URL provided in JSON data")
    
    try:
        # Download the image from the provided URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve image from URL")
        
        # Process the image directly from the byte stream
        image_data = BytesIO(response.content)  # Use BytesIO to handle the image in-memory
        image = Image.open(image_data)
        
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the image
        data = process_fingerprint(image, user_id)
        
        return JSONResponse(content={'success': True, 'data': data, 'message': 'Fingerprint processing completed'}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
#LOAD_TEST.PY

import httpx
import asyncio
import time
from asyncio import Semaphore

# Configuration
CONCURRENT_REQUESTS = 10  # Number of concurrent requests your server can handle
TOTAL_REQUESTS = 75      # Total number of requests to send

# Counters for tracking success and failure
success_count = 0
failure_count = 0

# Semaphore to limit concurrency
semaphore = Semaphore(CONCURRENT_REQUESTS)

async def fetch(client, url, user_id, image_url):
    global success_count, failure_count
    async with semaphore:  # Limit the number of concurrent requests
        try:
            response = await client.post(url, json={"userId": user_id, "finger": image_url})
            if response.status_code == 200:
                success_count += 1
            else:
                failure_count += 1
            print(f"Status Code: {response.status_code},")
        except httpx.RequestError as e:
            failure_count += 1
            print(f"Request error: {e}")

async def load_test():
    url = "http://127.0.0.1:8000/api/v1/fingerprints/extract"  # Corrected URL
    user_id = "test_user_id"
    image_url = "https://d1qeglde09f18h.cloudfront.net/photos/2024-07-25T02-02-35-440Z.jpeg"

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for _ in range(TOTAL_REQUESTS):
            task = asyncio.create_task(fetch(client, url, user_id, image_url))
            tasks.append(task)
        await asyncio.gather(*tasks)  # Wait for all tasks to complete

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(load_test())
    end_time = time.time()
    print(f"Load testing completed in {end_time - start_time:.2f} seconds")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {failure_count}")
    
    
    
    
    
    
    
    
#fingers_extract.py


from rembg import remove
from PIL import Image
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
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

        # Create a binary mask using the largest contour
        mask = np.zeros_like(mask)
        cv.drawContours(mask, [largest_contour], 0, 255, -1)

        # Perform post-processing to further improve the quality of the segmented region
        mask = cv.GaussianBlur(mask, (5, 5), 0)

        # Segment the hand or finger region from the rest of the image
        hand = cv.bitwise_and(img, img, mask=mask)

        # Convert to grayscale and apply adaptive thresholding
        hand_gray = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
        hand_thresh = cv.adaptiveThreshold(hand_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        # Find the bounding box of the largest contour
        x, y, w, h = cv.boundingRect(largest_contour)

        # Crop the image around the bounding box with some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1], x + w + 2 * padding)
        h = min(img.shape[0], y + h + 2 * padding)
        cropped_hand = hand_thresh[y:h, x:w]

        # Save the cropped segmented region image
        cv.imwrite(output_path, cropped_hand)

        # Display the results
        # plt.figure(figsize=(8, 8))
        # plt.subplot(1, 2, 1)
        # plt.title('Segmented Region')
        # plt.imshow(cropped_hand, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.title('Original Image with Mask')
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.show()
    else:
        print("No contours found!")







#NEW CODE
