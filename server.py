from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from flask_cors import CORS
from backend.db_helper import *
from main import *
import os
import sys
from datetime import datetime
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'  # Needed if you want to use sessions

# 1. Signup Route
@app.route('/signup_data', methods=['POST'])
def signup_data():
    data = request.get_json()
    if insert_signup(data['signupEmail'], data['username'], data['signupPassword']) == 1:
        return jsonify({'message': 'Data inserted successfully!'})
    else:
        return jsonify({'message': 'Error in inserting the Data!'})

# 2. Login Route
@app.route('/login_data', methods=['POST'])
def login_data():
    data = request.get_json()
    print(data)
    response_data = search_login_credentials(data['email'], data['password'])
    if response_data:
        return jsonify(response_data)
    return jsonify({'message': 'Data not found!'})

# 3. Index
@app.route('/')
def index_page():
    return render_template('index.html')

# 4. Verification Page
@app.route('/verify')
def verify():
    return render_template('verify.html')

# 5. Face Verification Logic
@app.route('/verify_face', methods=['POST'])
def verify_face():
    reference_image_path = "static/reference.jpg"

    if not os.path.exists(reference_image_path):
        return render_template('verify.html', error="Reference image not found.")

    # Load and encode reference image
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_encodings = face_recognition.face_encodings(reference_image)
    if not reference_encodings:
        return render_template('verify.html', error="No face found in reference image.")
    reference_encoding = reference_encodings[0]

    # Get uploaded image (captured from webcam)
    file = request.files.get('photo')
    if not file or file.filename == '':
        return render_template('verify.html', error="No image uploaded.")

    uploaded_image = face_recognition.load_image_file(file)
    uploaded_encodings = face_recognition.face_encodings(uploaded_image)
    if not uploaded_encodings:
        return render_template('verify.html', error="No face found in uploaded image.")
    uploaded_encoding = uploaded_encodings[0]

    # Compare face encodings
    is_match = face_recognition.compare_faces([reference_encoding], uploaded_encoding)[0]

    if is_match:
        return redirect(url_for('quiz_page'))
    else:
        return render_template('verify.html', error="Face verification failed. Please try again.")

# 6. Quiz Page
@app.route('/quiz_html')
def quiz_page():
    return render_template('quiz.html')

# 7. Video Feed
@app.route('/video_feed')
def video_feed():
    return Response(proctoringAlgo(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 8. Stop Camera
@app.route('/stop_camera')
def stop_camera():
    global running
    running = False
    main_app()
    print('Camera and Server stopping.....')
    os._exit(0)

# 9. Main
if __name__ == "__main__":
    print("Starting the Python Flask Server.....")
    app.run(port=5000, debug=True)
