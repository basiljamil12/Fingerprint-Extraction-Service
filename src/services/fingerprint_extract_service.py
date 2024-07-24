from src.utils.fingers_extract import process_image
import os
import cv2
import numpy as np

def process_fingerprint(file, user_id):
    temp_input_path = 'temp_input.png'
    temp_output_path = 'temp_output.png'
    
    file.save(temp_input_path)
    process_image(temp_input_path, temp_output_path)
    
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
