import cv2
import numpy as np

# Load the pre-trained Haar cascade classifier for car detection from the project directory
car_cascade = cv2.CascadeClassifier(r'/Users/samchou/Dubhack24/haarcascade_car.xml')

# Helper function to check if two line segments (p1-p2) and (q1-q2) intersect
def line_intersects(p1, p2, q1, q2):
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def on_segment(a, b, c):
        return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

    # Check the orientations needed to see if the segments intersect
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

# Function to detect lanes and determine if the car is crossing the lane
def detect_lanes_and_cars(frame):
    height, width, _ = frame.shape
    center_x = width // 2  # Car's assumed position (center of the frame)

    # Define x offset to shift the triangle horizontally
    x_offset = 120  # Positive value shifts to the right, negative shifts to the left

    # Define the "red zone" triangle size
    lane_crossing_threshold = 200  # Horizontal base of the triangle
    vertical_proximity_threshold = 150  # Height of the triangle
    move_down_offset = 90  # Move the triangle down

    bottom_threshold = int(2 * height // 3) + move_down_offset

    # Define the points for the triangular "red zone"
    triangle_points = np.array([
        [center_x - lane_crossing_threshold + x_offset, bottom_threshold],  # Left corner of the base
        [center_x + lane_crossing_threshold + x_offset, bottom_threshold],  # Right corner of the base
        [center_x + x_offset, bottom_threshold - vertical_proximity_threshold]  # Top point (apex)
    ]).astype(np.float32)

    # --- Create a separate layer for the filled triangle ---
    overlay = frame.copy()  # Create a copy of the frame for overlaying
    opacity = 0.4  # Set opacity level for the triangle

    # Draw the filled purple triangle on the overlay image
    cv2.fillPoly(overlay, [triangle_points.astype(np.int32)], color=(128, 0, 128))

    # Blend the overlay with the original frame to achieve the semi-transparent effect
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Draw the "red zone" triangle for debugging purposes (keeps the border)
    cv2.polylines(frame, [triangle_points.astype(np.int32)], isClosed=True, color=(128, 0, 128), thickness=3)

    # Define triangle sides as line segments
    tri_p1, tri_p2, tri_p3 = triangle_points
    triangle_sides = [(tri_p1, tri_p2), (tri_p2, tri_p3), (tri_p3, tri_p1)]

    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # **Region of Interest (ROI) for edge detection, but we don't alter the original frame**
    # Mask the top portion of the frame (trees, sky) and keep only the road area for processing
    mask = np.zeros_like(gray)
    polygon = np.array([[
        (0, height),  # bottom-left corner
        (width, height),  # bottom-right corner
        (width, int(height * 0.55)),  # top-right to cover 60% of the height
        (0, int(height * 0.55))  # top-left to cover 60% of the height
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    # Apply the mask to the edge detection step
    edges = cv2.Canny(gray, 150, 300)
    masked_edges = cv2.bitwise_and(edges, mask)

    # --- Add mask for the triangle area to exclude it from road line detection ---
    triangle_mask = np.zeros_like(gray)
    cv2.fillPoly(triangle_mask, [triangle_points.astype(np.int32)], 255)
    masked_edges = cv2.bitwise_and(masked_edges, cv2.bitwise_not(triangle_mask))  # Exclude triangle area

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

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

    # --- Restrict car detection area to be above the base of the triangle ---
    # Define a region of interest for car detection above the base of the triangle
    car_detection_area = frame[:int(bottom_threshold - vertical_proximity_threshold), :]  # Everything above the triangle

    # Convert this region to grayscale
    gray_car_detection_area = cv2.cvtColor(car_detection_area, cv2.COLOR_BGR2GRAY)

    # Detect cars using Haar Cascade within this restricted area
    cars = car_cascade.detectMultiScale(gray_car_detection_area, 1.1, 3)  # Adjust scaleFactor and minNeighbors for accuracy

    for (x, y, w, h) in cars:
        # Draw a rectangle around detected cars (adjust y-coordinates based on cropped area)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle around the car

    return frame

# Load video
#cap = cv2.VideoCapture(r'/Users/samchou/Dubhack24/lane_video.mp4')
cap = cv2.VideoCapture(r'/Users/samchou/Dubhack24/brakecheck_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes and cars
    lanes_frame = detect_lanes_and_cars(frame)

    # Display the result
    cv2.imshow('Lane and Car Detection', lanes_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
