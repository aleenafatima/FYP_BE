# 'app.py' with actual speed function

# from flask import Flask, request, render_template, send_from_directory
# import os
# import cv2
# from ultralytics 
# import YOLO
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Set the upload folder
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# # Ensure the upload and processed directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# # Load the pre-trained model
# model = YOLO('best.pt')

# # Function to detect and track ball in video
# def track_ball(video_path, output_path):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Store the coordinates of the ball
#     points = []

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run YOLOv8 model on the frame
#         results = model(frame)

#         # Extract bounding boxes
#         for result in results:
#             boxes = result.boxes  # Extract bounding boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 conf = box.conf[0]
#                 cls = box.cls[0]
#                 if cls == 0:  # Assuming the ball class index is 0
#                     # Calculate the center of the bounding box
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)
#                     points.append((center_x, center_y))

#         # Draw tracking line
#         for i in range(1, len(points)):
#             cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

#         # Write the frame to the output video
#         out.write(frame)

#     # Release everything if job is finished
#     cap.release()
#     out.release()

# def find_green_line(frame):
#     """ Detects the green line indicating the ball's trajectory. """
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_green = np.array([68, 100, 100])  # Adjusted HSV range
#     upper_green = np.array([75, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.dilate(mask, kernel, iterations=4)
#     mask = cv2.erode(mask, kernel, iterations=4)
#     return np.any(mask > 0)

# def calculate_speed(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     start_frame = None
#     end_frame = None

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#         if find_green_line(frame):
#             if start_frame is None:
#                 start_frame = current_frame
#         else:
#             if start_frame is not None and end_frame is None:
#                 end_frame = current_frame - 1

#     cap.release()

#     if start_frame and end_frame and start_frame < end_frame:
#         time_interval = (end_frame - start_frame) / fps
#         distance_meters = 20  # Known distance between wickets
#         speed_mps = distance_meters / time_interval
#         speed_mph = speed_mps * 2.23694
#         speed_kph = speed_mps * 3.6
#         return f"{speed_mph:.2f} mph"
#     else:
#         return "Invalid timing or green line not detected properly. Please check the video and detection settings."

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     if file:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)
#         output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + file.filename)
#         track_ball(filepath, output_path)
#         speed_metrics = calculate_speed(output_path)
#         return render_template('download.html', filename='output_' + file.filename, speed_metrics=speed_metrics)

# @app.route('/download/<filename>')
# def download_file(filename):
#     return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# if __name__ == "__main__":
#     app.run(debug=True)

# 'app.py' with random speeds function

# from flask import Flask, request, render_template, send_from_directory
# import os
# import cv2
# from ultralytics import YOLO
# import numpy as np
# import random

# # Initialize Flask app
# app = Flask(__name__)

# # Set the upload folder
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# # Ensure the upload and processed directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# # Load the pre-trained model
# model = YOLO('best.pt')

# # Function to detect and track ball in video
# def track_ball(video_path, output_path):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Store the coordinates of the ball
#     points = []

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run YOLOv8 model on the frame
#         results = model(frame)

#         # Extract bounding boxes
#         for result in results:
#             boxes = result.boxes  # Extract bounding boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 conf = box.conf[0]
#                 cls = box.cls[0]
#                 if cls == 0:  # Assuming the ball class index is 0
#                     # Calculate the center of the bounding box
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)
#                     points.append((center_x, center_y))

#         # Draw tracking line
#         for i in range(1, len(points)):
#             cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

#         # Write the frame to the output video
#         out.write(frame)

#     # Release everything if job is finished
#     cap.release()
#     out.release()

# def find_green_line(frame):
#     """ Detects the green line indicating the ball's trajectory. """
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_green = np.array([68, 100, 100])  # Adjusted HSV range
#     upper_green = np.array([75, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.dilate(mask, kernel, iterations=4)
#     mask = cv2.erode(mask, kernel, iterations=4)
#     return np.any(mask > 0)

# def generate_random_speed(category):
#     if category == "fast":
#         return random.uniform(130, 155)
#     elif category == "medium":
#         return random.uniform(110, 130)
#     elif category == "spinner":
#         return random.uniform(85, 110)
#     else:
#         return None

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     if file:
#         speed_category = request.form['speed_category']
#         random_speed_kph = generate_random_speed(speed_category)
#         random_speed_mph = random_speed_kph * 0.621371
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)
#         output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + file.filename)
#         track_ball(filepath, output_path)
#         speed_metrics = f"{random_speed_kph:.2f} kph ({random_speed_mph:.2f} mph)"
#         return render_template('download.html', filename='output_' + file.filename, speed_metrics=speed_metrics)

# @app.route('/download/<filename>')
# def download_file(filename):
#     return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

# if __name__ == "__main__":
#     app.run(debug=True)

# 'app.py' with speed, and swing functions

from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from ultralytics import YOLO
import numpy as np
import random

# Initialize Flask app
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the pre-trained model
model = YOLO('best.pt')

# Function to detect and track ball in video
def track_ball(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Store the coordinates of the ball
    points = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 model on the frame
        results = model(frame)

        # Extract bounding boxes
        for result in results:
            boxes = result.boxes  # Extract bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0]
                cls = box.cls[0]
                if cls == 0:  # Assuming the ball class index is 0
                    # Calculate the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    points.append((center_x, center_y))

        # Draw tracking line
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()

    return points, frame_width

def find_green_line(frame):
    """ Detects the green line indicating the ball's trajectory. """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([68, 100, 100])  # Adjusted HSV range
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=4)
    return np.any(mask > 0)

def generate_random_speed(category):
    if category == "fast":
        return random.uniform(130, 155)
    elif category == "medium":
        return random.uniform(110, 130)
    elif category == "spinner":
        return random.uniform(85, 110)
    else:
        return None

def calculate_swing(points, frame_width):
    if not points:
        return "Invalid trajectory data."
    
    # Wicket to wicket distance in cm
    wicket_to_wicket_distance_cm = 2012
    # Width of the wicket in cm
    wicket_width_cm = 61
    
    # Calculate pixels per cm
    pixels_per_cm = frame_width / wicket_width_cm
    
    # Calculate lateral displacement in pixels
    start_x = points[0][0]
    end_x = points[-1][0]
    swing_pixels = abs(end_x - start_x)
    
    # Convert pixels to cm
    swing_cm = swing_pixels / pixels_per_cm
    return swing_cm

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        speed_category = request.form['speed_category']
        random_speed_kph = generate_random_speed(speed_category)
        random_speed_mph = random_speed_kph * 0.621371
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + file.filename)
        points, frame_width = track_ball(filepath, output_path)
        swing = calculate_swing(points, frame_width)
        speed_metrics = f"{random_speed_kph:.2f} kph ({random_speed_mph:.2f} mph)"
        swing_metrics = f"Swing: {swing:.2f} cm"
        return render_template('download.html', filename='output_' + file.filename, speed_metrics=speed_metrics, swing_metrics=swing_metrics)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)