# from flask import request, jsonify
# from src.services.face_recognition_service import recognition_service
# import numpy as np
# import cv2

# def extract_fingerprints():
#     if 'finger' not in request.files:
#         return jsonify({'success': False, 'data': {}, 'message': 'No image file provided'}), 400

#     uploaded_file = request.files['finger']
#     course_id = request.form.get('courseId')

#     if not uploaded_file:
#         return jsonify({'success': False, 'data': {}, 'message': 'No finger provided in form data'}), 400
    
#     if not course_id:
#         return jsonify({'success': False, 'data': {}, 'message': 'No courseId provided in form data'}), 400
    
#     data = recognition_service(uploaded_file, course_id)
    
#     formatted_data = {
#     'attendance_records': data[0],
#     'detected_students_in_image': data[1],
#     'no_of_niqaabi_students_in_image': data[2],
#     'detected_image': data[3],
#     'recognized_students_in_image': data[4],
#     'unrecognized_students_in_image': data[5]
#     }

#     return jsonify({'success': True, 'data': formatted_data, 'message': 'Face recognition completed'}), 200
from flask import request, jsonify
from src.services.fingerprint_extract_service import process_fingerprint

def extract_fingerprints():
    if 'finger' not in request.files:
        return jsonify({'success': False, 'data': {}, 'message': 'No image file provided'}), 400
    
    user_id = request.form.get('userId')
    uploaded_file = request.files['finger']
    

    if not uploaded_file:
        return jsonify({'success': False, 'data': {}, 'message': 'No finger provided in form data'}), 400
    
    if not user_id:
        return jsonify({'success': False, 'data': {}, 'message': 'No courseId provided in form data'}), 400
    
    try:
        data = process_fingerprint(uploaded_file, user_id)
        return jsonify({'success': True, 'data': data, 'message': 'Fingerprint processing completed'}), 200
    except Exception as e:
        return jsonify({'success': False, 'data': {}, 'message': str(e)}), 500
