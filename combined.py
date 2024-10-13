import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' for better accuracy but it's slower

# Open the video file
cap = cv2.VideoCapture('websiteclip.mov')

# Get video properties to calculate frame delay
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_delay = int(400 / fps)  # Calculate delay between frames in milliseconds (default speed)

def estimate_distance(box_height, frame_height, scale_factor):
    distance = (frame_height / (box_height + 1e-5)) * scale_factor  # Avoid division by zero
    return round(distance, 2)

def line_intersects(p1, p2, q1, q2):
    # Helper function to compute orientation
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise

    def on_segment(a, b, c):
        if min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1]):
            return True
        return False

    # Calculate the orientations needed to check intersection
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases (collinear points)
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

# Function to detect lanes and determine if a car is crossing the lane
def detect_lanes(frame):
    height, width, _ = frame.shape
    center_x = width // 2  # Car's assumed position (center of the frame)

    # Adjustments: Shift the triangle to the left, closer to the bottom, and shrink it
    x_offset = 50  # Shift triangle further to the left
    lane_crossing_threshold = 200  # Shrink the horizontal base of the triangle
    vertical_proximity_threshold = 100  # Shrink the height of the triangle
    move_down_offset = 200  # Move the triangle down, closer to the bottom of the screen

    bottom_threshold = int(3 * height // 4) + move_down_offset  # Adjusted position of the triangle base

    # Define the points for the triangular red zone (isosceles triangle) with the new parameters
    triangle_points = np.array([
        [center_x - lane_crossing_threshold + x_offset, bottom_threshold],  # Left corner of the base
        [center_x + lane_crossing_threshold + x_offset, bottom_threshold],  # Right corner of the base
        [center_x + x_offset, bottom_threshold - vertical_proximity_threshold]  # Top point (apex)
    ]).astype(np.float32)

    # Define triangle sides as line segments
    tri_p1, tri_p2, tri_p3 = triangle_points
    triangle_sides = [(tri_p1, tri_p2), (tri_p2, tri_p3), (tri_p3, tri_p1)]

    # Draw the triangle for visualization
    #cv2.polylines(frame, [np.array(triangle_points, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Cropping the frame to remove the top part and also ignore the bottom 5%
    crop_y_start = int(height // 2)  # Crop the top half of the frame
    crop_y_end = int(1 * height)  # Ignore the bottom 5% of the frame
    cropped_frame = frame[crop_y_start:crop_y_end, :]  # Crop the frame vertically, keeping the full width

    # Convert the cropped frame to grayscale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 150, 300)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Adjust y-coordinates because we cropped part of the frame
            y1 += crop_y_start
            y2 += crop_y_start

            # Convert points to float for intersection testing
            point1 = (float(x1), float(y1))
            point2 = (float(x2), float(y2))

            # Check if the line intersects with any of the triangle's sides
            intersects_triangle = any(line_intersects(point1, point2, *side) for side in triangle_sides)

            # If the line intersects the triangle, color it red
            if intersects_triangle:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red line
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green line

    return frame


# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties (optional, in case you want to adjust threshold dynamically)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set a threshold to ignore objects that are detected too close to the bottom of the frame
DASHBOARD_THRESHOLD = int(0.95 * height)  # Ignore objects if the bottom Y-coordinate is within 95% of the frame height

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

    lanes_frame = detect_lanes(frame)

    # Display the result
    cv2.imshow('Lane Detection', lanes_frame)

    # Run YOLOv8 model on the frame
    results = model(frame)

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
                    continue  # Skip this detection, it's likely part of the dashboard

                # Get the scale factor based on the class (car, bus, or truck)
                scale_factor = SCALE_FACTORS.get(cls, 1.0)  # Default to 1.0 if not found

                # Estimate distance from the car, bus, or truck
                distance = estimate_distance(box_height, height, scale_factor)

                # Set a color for the bounding box
                color = (0, 255, 0)  # Green for safe cars

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Create label with class name and estimated distance
                label = f"{model.names[cls]} {distance}m"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame in a window at 2x speed (half the usual wait time)
    cv2.imshow('Processed Frame', frame)

    # Wait for half the normal frame delay for 2x speed
    if cv2.waitKey(frame_delay // 2) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
