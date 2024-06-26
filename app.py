from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from ultralytics import YOLO

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + file.filename)
        track_ball(filepath, output_path)
        return render_template('download.html', filename='output_' + file.filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)