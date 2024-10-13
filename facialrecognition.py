import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import os
import time
from collections import deque
import torch

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the .dat file
dat_file_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

# Load face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(dat_file_path)

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set device to GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Smartphone COCO class ID is 67, change this if needed
SMARTPHONE_CLASS_ID = 67

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def gaze_ratio(eye_points, facial_landmarks, frame, gray):
    try:
        left_eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points])
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        
        gray_eye = eye[min_y: max_y, min_x: max_x]
        
        if gray_eye.size == 0:
            return 1  # Default value if eye region is empty
        
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        
        if threshold_eye is None or threshold_eye.size == 0:
            return 1  # Default value if thresholding fails
        
        height, width = threshold_eye.shape
        if width == 0:
            return 1  # Avoid division by zero
        
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        
        if left_side_white == 0 and right_side_white == 0:
            return 1  # Default value if no white pixels detected
        elif left_side_white == 0:
            return 5  # Looking left
        elif right_side_white == 0:
            return 0  # Looking right
        else:
            return left_side_white / right_side_white
    except Exception as e:
        print(f"Error in gaze_ratio: {str(e)}")
        return 1  # Default value on error

# Initialize a deque to store the last 5 seconds of eye contact status
eye_contact_history = deque(maxlen=5)

def detect_eye_contact(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        
        # Get the eye regions
        left_eye = [36, 37, 38, 39, 40, 41]
        right_eye = [42, 43, 44, 45, 46, 47]
        
        left_eye_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in left_eye])
        right_eye_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in right_eye])
        
        # Calculate eye aspect ratio
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate gaze ratio
        gaze_ratio_left = gaze_ratio(left_eye, landmarks, frame, gray)
        gaze_ratio_right = gaze_ratio(right_eye, landmarks, frame, gray)
        gaze_ratio_avg = (gaze_ratio_left + gaze_ratio_right) / 2
        
        # Determine if eyes are open and looking at the camera
        if ear > 0.2 and 0.8 < gaze_ratio_avg < 2.0:
            eye_contact_history.append(1)  # 1 represents making eye contact
        else:
            eye_contact_history.append(0)  # 0 represents not making eye contact

        # Calculate the number of seconds without eye contact in the last 5 seconds
        no_eye_contact_count = eye_contact_history.count(0)

        # Choose color based on eye contact history
        color = (0, 255, 0) if no_eye_contact_count < 3 else (0, 0, 255)
        
        # Draw rectangles around eyes
        for eye in [left_eye_pts, right_eye_pts]:
            hull = cv2.convexHull(eye)
            cv2.drawContours(frame, [hull], -1, color, 1)

        # Draw outline around the entire head
        jaw_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        jaw_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in jaw_points])
        cv2.polylines(frame, [jaw_pts], False, color, 2)

        # Draw outline around the nose
        nose_points = [27, 28, 29, 30, 31, 32, 33, 34, 35]
        nose_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in nose_points])
        cv2.polylines(frame, [nose_pts], False, color, 2)

        # Draw circle around the chin
        chin_x = landmarks.part(8).x
        chin_y = landmarks.part(8).y
        cv2.circle(frame, (chin_x, chin_y), 10, color, 2)

    return frame

def detect_smartphone(frame):
    # Convert frame to RGB (YOLO expects RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model(img_rgb)
    
    # Parse results
    detected_objects = results.pandas().xyxy[0]
    
    for i, obj in detected_objects.iterrows():
        if obj['class'] == SMARTPHONE_CLASS_ID:
            # Get bounding box coordinates
            x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            label = f"{obj['name']} {obj['confidence']:.2f}"
            
            # Draw bounding box around smartphone in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text
    
    return frame

def analyze_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object to save the output video
    output_path = 'output_' + os.path.basename(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analyze eye contact
        frame_with_eye_contact = detect_eye_contact(frame)
        
        # Detect smartphone
        frame_with_detections = detect_smartphone(frame_with_eye_contact)

        # Write the frame to the output video
        out.write(frame_with_detections)

        # Display the frame (optional, for real-time viewing)
        cv2.imshow('Eye Contact and Smartphone Detection', frame_with_detections)

        # Calculate and print the processing speed
        if frame_count % fps == 0:  # Print every second
            elapsed_time = time.time() - start_time
            print(f"Processing speed: {frame_count / elapsed_time:.2f} fps")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Analysis complete. Output saved to {output_path}")

# Usage
video_path = 'eyesclosed.MOV'  # Replace with your video file path
analyze_video(video_path)