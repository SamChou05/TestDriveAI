import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' for better accuracy but it's slower

# Open the video file
cap = cv2.VideoCapture('input_video2.mp4')

# Function to estimate distance based on bounding box height and vehicle type
def estimate_distance(box_height, frame_height, scale_factor):
    distance = (frame_height / (box_height + 1e-5)) * scale_factor  # Avoid division by zero
    return round(distance, 2)

# Function to check if any point within the bounding box falls inside the trapezoid
def is_in_trapezoid(box, trapezoid_pts):
    x1, y1, x2, y2 = box  # Unpack bounding box coordinates
    step_size = 5  # Adjust the step size for denser or sparser grid (smaller step -> more points checked)
    
    for x in range(x1, x2, step_size):
        for y in range(y1, y2, step_size):
            if cv2.pointPolygonTest(np.array(trapezoid_pts, np.int32), (float(x), float(y)), False) >= 0:
                return True
    return False

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties (optional, in case you want to adjust threshold dynamically)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set thresholds to ignore objects that are detected too close to the bottom of the frame
DASHBOARD_THRESHOLD = int(0.85 * height)  # Ignore detections in the bottom 15% of the frame

# Scaling factors for different types of vehicles to adjust distance estimation
SCALE_FACTORS = {
    2: 1.0,  # Car: baseline distance scale
    5: 2.5,  # Bus: bigger, so we scale the distance estimation by a larger factor
    7: 1.7   # Truck: larger than a car but smaller than a bus
}

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the frame
    results = model(frame)

    closest_box_center = None

    # Subject position (bottom-center of the frame)
    subject_position = (width // 2, height)

    # Define the shorter trapezoid danger zone with adjusted Y-coordinates
    trapezoid_top_left = (int(0.4 * width), int(0.85 * height))  # Adjust top left point of trapezoid
    trapezoid_top_right = (int(0.6 * width), int(0.85 * height))  # Adjust top right point of trapezoid
    trapezoid_bottom_left = (int(0.45 * width), int(0.74 * height))  # Adjust bottom left point to 65% height
    trapezoid_bottom_right = (int(0.55 * width), int(0.74 * height))  # Adjust bottom right point to 65% height
    trapezoid_pts_outer = [trapezoid_top_left, trapezoid_top_right, trapezoid_bottom_right, trapezoid_bottom_left]

    inner_trapezoid_top_left = (int(0.42 * width), int(0.85 * height))  # Top left point of inner trapezoid
    inner_trapezoid_top_right = (int(0.58 * width), int(0.85 * height))  # Top right point of inner trapezoid
    inner_trapezoid_bottom_left = (int(0.47 * width), int(0.79 * height))  # Bottom left point of inner trapezoid
    inner_trapezoid_bottom_right = (int(0.53 * width), int(0.79 * height))  # Bottom right point of inner trapezoid
    trapezoid_pts_inner = [inner_trapezoid_top_left, inner_trapezoid_top_right, inner_trapezoid_bottom_right, inner_trapezoid_bottom_left]

    # Comment out the drawing of the trapezoid for visualization
    # cv2.polylines(frame, [np.array(trapezoid_pts_outer, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    # cv2.polylines(frame, [np.array(trapezoid_pts_inner, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Loop over detections and draw boxes
    for result in results:
        # Extract boxes and classes
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class indices
        for i in range(len(boxes)):
            cls = int(classes[i])
            # Check if the detected object is a car, bus, or truck
            if cls in [2, 5, 7]:  # COCO classes: 2=car, 5=bus, 7=truck
                x1, y1, x2, y2 = boxes[i].astype(int)
                box_height = y2 - y1

                # Ignore detections where the bottom of the bounding box is too close to the bottom of the frame
                if y2 > DASHBOARD_THRESHOLD:
                    continue  # Skip this detection, it's in the bottom 15% of the frame

                # Get the scale factor based on the class (car, bus, or truck)
                scale_factor = SCALE_FACTORS.get(cls, 1.0)  # Default to 1.0 if not found

                # Estimate distance from the car, bus, or truck
                distance = estimate_distance(box_height, height, scale_factor)

                # Check if any part of the bounding box is inside the trapezoid
                if is_in_trapezoid((x1, y1, x2, y2), trapezoid_pts_inner):
                    color = (0, 0, 255)  # Red for very close cars in the inner trapezoid
                elif is_in_trapezoid((x1, y1, x2, y2), trapezoid_pts_outer):
                    color = (0, 255, 255)  # Yellow for cars in the outer trapezoid
                else:
                    color = (0, 255, 0)  # Green for safe cars

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Create label with class name and estimated distance
                label = f"{model.names[cls]} {distance}m"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame in a window
    cv2.imshow('Processed Frame', frame)
    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
