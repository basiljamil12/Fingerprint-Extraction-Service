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

