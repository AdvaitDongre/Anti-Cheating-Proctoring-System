import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def enhance_contrast(frame):
    """Enhance image contrast using CLAHE for better performance in low light"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def get_gaze_direction(frame, face_landmarks, frame_width, frame_height):
    """Calculate gaze direction based on iris positions"""
    # Iris landmarks (MediaPipe iris centers)
    RIGHT_IRIS = [468]
    LEFT_IRIS = [473]
    
    # Get eye contours
    right_eye_contour = list(set([idx for connection in mp_face_mesh.FACEMESH_RIGHT_EYE for idx in connection]))
    left_eye_contour = list(set([idx for connection in mp_face_mesh.FACEMESH_LEFT_EYE for idx in connection]))

    # Process right eye
    right_iris = face_landmarks.landmark[RIGHT_IRIS[0]]
    right_iris_x = int(right_iris.x * frame_width)
    right_iris_y = int(right_iris.y * frame_height)
    
    # Process left eye
    left_iris = face_landmarks.landmark[LEFT_IRIS[0]]
    left_iris_x = int(left_iris.x * frame_width)
    left_iris_y = int(left_iris.y * frame_height)

    # Calculate eye regions
    def get_eye_region(contour_indices):
        x_coords = [int(face_landmarks.landmark[i].x * frame_width) for i in contour_indices]
        y_coords = [int(face_landmarks.landmark[i].y * frame_height) for i in contour_indices]
        return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))

    # Get eye regions
    r_min_x, r_max_x, r_min_y, r_max_y = get_eye_region(right_eye_contour)
    l_min_x, l_max_x, l_min_y, l_max_y = get_eye_region(left_eye_contour)

    # Calculate horizontal ratios
    right_h_ratio = (right_iris_x - r_min_x) / (r_max_x - r_min_x) if (r_max_x - r_min_x) != 0 else 0.5
    left_h_ratio = (left_iris_x - l_min_x) / (l_max_x - l_min_x) if (l_max_x - l_min_x) != 0 else 0.5

    # Determine horizontal direction
    h_direction = "center"
    if right_h_ratio < 0.4 and left_h_ratio > 0.6:
        h_direction = "right"
    elif right_h_ratio > 0.6 and left_h_ratio < 0.4:
        h_direction = "left"

    # Calculate vertical ratios
    right_v_ratio = (right_iris_y - r_min_y) / (r_max_y - r_min_y) if (r_max_y - r_min_y) != 0 else 0.5
    left_v_ratio = (left_iris_y - l_min_y) / (l_max_y - l_min_y) if (l_max_y - l_min_y) != 0 else 0.5

    # Determine vertical direction
    v_direction = "center"
    if right_v_ratio < 0.4 and left_v_ratio < 0.4:
        v_direction = "up"
    elif right_v_ratio > 0.6 and left_v_ratio > 0.6:
        v_direction = "down"

    # Combine directions
    directions = []
    if h_direction != "center":
        directions.append(h_direction)
    if v_direction != "center":
        directions.append(v_direction)
    
    return ' '.join(directions) if directions else "center", (right_iris_x, right_iris_y), (left_iris_x, left_iris_y)

def process_frame(frame):
    """Process each frame for gaze detection"""
    frame = cv2.flip(frame, 1)  # Mirror frame
    enhanced_frame = enhance_contrast(frame)
    results = face_mesh.process(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
    frame_height, frame_width = frame.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            direction, right_pupil, left_pupil = get_gaze_direction(
                frame, face_landmarks, frame_width, frame_height)

            # Draw pupils
            cv2.circle(frame, right_pupil, 3, (0, 0, 255), -1)
            cv2.circle(frame, left_pupil, 3, (0, 0, 255), -1)
            
            # Display direction
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Looking at screen" if direction == "center" else "Looking away!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if direction != "center" else (0, 255, 0), 2)

    return frame

def gaze_detection(source=0):
    """Main function for gaze detection"""
    if isinstance(source, str):
        if source.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image processing
            frame = cv2.imread(source)
            if frame is not None:
                processed = process_frame(frame)
                cv2.imshow('Gaze Detection', processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return
        else:
            # Video processing
            cap = cv2.VideoCapture(source)
    else:
        # Webcam processing
        cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Gaze Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use webcam (0) by default, or specify file path
    gaze_detection(0)  # Replace 0 with "path/to/file" for image/video