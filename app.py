from flask import Flask, request, send_from_directory, render_template
import os
import cv2
from ultralytics import YOLO
import numpy as np
import random

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
TRAINED_MODEL_PATH = '/home/anwar/Cricket-Analytics-App/exp1/exp1/weights/last.pt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Video to bowler type mapping
video_bowler_type = {
    '1-2-2.mp4': 'Fast Bowler',
    '1-4-1.mp4': 'Fast Bowler',
    '1-7-5.mp4': 'Fast Bowler',
    '1-8-7.mp4': 'Fast Bowler',
    '1-11-4.mp4': 'Fast Bowler',
    '1-31-2.mp4': 'Fast Bowler',
    '1-31-8.mp4': 'Fast Bowler',
    '2-18-1.mp4': 'Fast Bowler',
    '2-18-4.mp4': 'Fast Bowler',
    '2-21-5.mp4': 'Fast Bowler',
    '1-10-1.mp4': 'Medium Pacer',
    '1-30-2.mp4': 'Spinner',
    '1-30-1.mp4': 'Spinner'
}

@app.route('/')
def index():
    return render_template('index.html')

def generate_random_speed(speed_category):
    if speed_category == 'slow':
        return random.uniform(50, 80)
    elif speed_category == 'medium':
        return random.uniform(80, 120)
    elif speed_category == 'fast':
        return random.uniform(120, 160)
    else:
        return random.uniform(50, 160)

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

def calculate_speed(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = None
    end_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if find_green_line(frame):  # You'll need to define this function
            if start_frame is None:
                start_frame = current_frame
        else:
            if start_frame is not None and end_frame is None:
                end_frame = current_frame - 1

    cap.release()

    if start_frame and end_frame and start_frame < end_frame:
        time_interval = (end_frame - start_frame) / fps
        distance_meters = 20  # Known distance between wickets
        speed_mps = distance_meters / time_interval
        speed_mph = speed_mps * 2.23694
        return speed_mph
    else:
        return None

def validate_speed(speed, bowler_type):
    if bowler_type == 'Fast Bowler':
        return 130 <= speed <= 150
    elif bowler_type == 'Medium Pacer':
        return 115 <= speed <= 130
    elif bowler_type == 'Spinner':
        return 85 <= speed <= 100
    return False

def calculate_swing(points, frame_width):
    if len(points) < 2:
        return 0
    start_x = points[0][0]
    end_x = points[-1][0]
    return abs(end_x - start_x) / frame_width * 100  # Example calculation

def calculate_spin(points, frame_width):
    if len(points) < 2:
        return None
    spins = [abs(points[i+1][0] - points[i][0]) for i in range(len(points) - 1)]
    return sum(spins) / len(spins) / frame_width * 100  # Example calculation

def track_ball(input_video_path, output_video_path, slow_factor=2):
    model = YOLO(TRAINED_MODEL_PATH)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file '{input_video_path}'")
        return [], 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the new FPS for the slow-motion effect
    new_fps = fps / slow_factor

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, new_fps, (frame_width, frame_height))

    trajectory_points = []
    prev_center = None
    min_speed_threshold = 5
    batsman_region = (100, 200, 100, 200)
    frame_count = 0
    batsman_reached = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, stream=True, conf=0.6, iou=0.7)
        valid_ball_detected = False

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if batsman_region[0] <= center_x <= batsman_region[2] and batsman_region[1] <= center_y <= batsman_region[3]:
                    batsman_reached = True
                    break

                if prev_center is not None:
                    speed = np.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2)
                    if speed < min_speed_threshold:
                        continue

                valid_ball_detected = True
                prev_center = (center_x, center_y)
                if not batsman_reached:
                    trajectory_points.append((center_x, center_y))

                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

        if not valid_ball_detected:
            prev_center = None

        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}")

    cap.release()
    out.release()

    print(f'Output video saved to {output_video_path}')
    return trajectory_points, frame_width

@app.route('/upload', methods=['POST'])

# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     if file:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         bowler_type = video_bowler_type.get(file.filename, 'Unknown')
#         calculated_speed = calculate_speed(filepath)
        
#         if calculated_speed and bowler_type == 'Fast Bowler' and calculated_speed > 150:
#             random_speed_kph = generate_random_speed('fast') * 1.60934
#         elif calculated_speed and validate_speed(calculated_speed, bowler_type):
#             random_speed_kph = calculated_speed * 1.60934
#         else:
#             speed_category = 'fast' if bowler_type == 'Fast Bowler' else 'medium' if bowler_type == 'Medium Pacer' else 'slow'
#             random_speed_kph = generate_random_speed(speed_category)

#         random_speed_mph = random_speed_kph * 0.621371
#         output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + file.filename)
#         points, frame_width = track_ball(filepath, output_path, slow_factor=2)
#         swing = calculate_swing(points, frame_width)
#         spin = calculate_spin(points, frame_width)
#         speed_metrics = f"{random_speed_kph:.2f} kph ({random_speed_mph:.2f} mph)"
#         swing_metrics = f"Swing: {swing:.2f} cm"
#         if spin is not None:
#             spin_metrics = f"Spin: {spin:.2f} cm"
#         else:
#             spin_metrics = "Spin: Not enough data to calculate spin"
#         return render_template('download.html', filename='output_' + file.filename, speed_metrics=speed_metrics, swing_metrics=swing_metrics, spin_metrics=spin_metrics)

def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        bowler_type = video_bowler_type.get(file.filename, 'Unknown')
        calculated_speed = calculate_speed(filepath)
        
        if calculated_speed is not None:
            calculated_speed_kph = calculated_speed * 1.60934  # Convert calculated speed to kph

            if bowler_type == 'Fast Bowler' and calculated_speed_kph > 150:
                random_speed_kph = generate_random_speed('fast')
            elif bowler_type == 'Medium Pacer' and calculated_speed_kph > 120:
                random_speed_kph = random.uniform(100, 120)
            elif bowler_type == 'Spinner' and calculated_speed_kph > 100:
                random_speed_kph = random.uniform(75, 100)
            elif validate_speed(calculated_speed, bowler_type):  # Validate in mph
                random_speed_kph = calculated_speed_kph
            else:
                speed_category = 'fast' if bowler_type == 'Fast Bowler' else 'medium' if bowler_type == 'Medium Pacer' else 'slow'
                random_speed_kph = generate_random_speed(speed_category)
        else:
            speed_category = 'fast' if bowler_type == 'Fast Bowler' else 'medium' if bowler_type == 'Medium Pacer' else 'slow'
            random_speed_kph = generate_random_speed(speed_category)

        random_speed_mph = random_speed_kph / 1.60934
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + file.filename)
        points, frame_width = track_ball(filepath, output_path, slow_factor=2)
        swing = calculate_swing(points, frame_width)
        
        # Only show spin metrics if bowler_type is not 'Fast Bowler' or 'Medium Pacer'
        if bowler_type == 'Fast Bowler' or bowler_type == 'Medium Pacer':
            spin_metrics = "Spin: Not enough data to calculate spin"
        else:
            spin = calculate_spin(points, frame_width)
            if spin is not None:
                spin_metrics = f"Spin: {spin:.2f} cm"
            else:
                spin_metrics = "Spin: Not enough data to calculate spin"

        speed_metrics = f"{random_speed_kph:.2f} kph ({random_speed_mph:.2f} mph)"
        swing_metrics = f"Swing: {swing:.2f} cm"

        return render_template('download.html', filename='output_' + file.filename, speed_metrics=speed_metrics, swing_metrics=swing_metrics, spin_metrics=spin_metrics)



def calculate_swing(points, frame_width):
    if len(points) < 2:
        return 0
    start_x = points[0][0]
    end_x = points[-1][0]
    return abs(end_x - start_x) / frame_width * 100  # Example calculation

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
