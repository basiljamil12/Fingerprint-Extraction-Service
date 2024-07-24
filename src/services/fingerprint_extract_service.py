from src.utils.fingers_extract import process_image
import os
import cv2
import numpy as np
import uuid
from datetime import datetime

def process_fingerprint(file, user_id):
    # Generate unique file names using timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex
    temp_input_path = f'temp_input_{timestamp}_{unique_id}.png'
    temp_output_path = f'temp_output_{timestamp}_{unique_id}.png'
    
    # Save the uploaded file
    file.save(temp_input_path)
    
    # Process the image
    process_image(temp_input_path, temp_output_path)
    
    # Read the processed image
    processed_image = cv2.imread(temp_output_path, cv2.IMREAD_GRAYSCALE)
    processed_image_bytes = cv2.imencode('.png', processed_image)[1].tobytes()
    
    # Clean up temporary files
    os.remove(temp_input_path)
    os.remove(temp_output_path)
    
    # Prepare data to be returned
    data = {
        'user_id': user_id,
        'processed_image': processed_image_bytes.hex()  # convert to hex string
    }
    
    return data
