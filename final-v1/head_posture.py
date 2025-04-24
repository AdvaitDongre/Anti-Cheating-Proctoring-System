import cv2
import mediapipe as mp
import numpy as np
import math
import time
import argparse

# Constants for head pose estimation
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_EAR = 234
RIGHT_EAR = 454
NOSE_TIP = 4
FOREHEAD = 10
CHIN = 152

def calculate_head_pose(image, landmarks):
        img_h, img_w, _ = image.shape
        
        # Convert landmarks to pixel coordinates
        nose_2d = (landmarks[NOSE_TIP].x * img_w, landmarks[NOSE_TIP].y * img_h)
        forehead_2d = (landmarks[FOREHEAD].x * img_w, landmarks[FOREHEAD].y * img_h)
        chin_2d = (landmarks[CHIN].x * img_w, landmarks[CHIN].y * img_h)
        left_ear_2d = (landmarks[LEFT_EAR].x * img_w, landmarks[LEFT_EAR].y * img_h)
        right_ear_2d = (landmarks[RIGHT_EAR].x * img_w, landmarks[RIGHT_EAR].y * img_h)
        
        # Calculate 3D approximation
        face_width = np.linalg.norm(np.array(left_ear_2d) - np.array(right_ear_2d))
        face_height = np.linalg.norm(np.array(forehead_2d) - np.array(chin_2d))
        
        # Calculate vertical tilt
        chin_to_nose = np.array([nose_2d[0] - chin_2d[0], nose_2d[1] - chin_2d[1]])
        vertical_ref = np.array([0, -1])
        
        chin_to_nose_norm = chin_to_nose / np.linalg.norm(chin_to_nose) if np.linalg.norm(chin_to_nose) > 0 else chin_to_nose
        vertical_ref_norm = vertical_ref / np.linalg.norm(vertical_ref)
        
        dot_product = np.clip(np.dot(chin_to_nose_norm, vertical_ref_norm), -1.0, 1.0)
        vertical_angle = np.degrees(np.arccos(dot_product))
        
        # Determine direction
        forehead_nose_y_diff = forehead_2d[1] - nose_2d[1]
        chin_nose_y_diff = chin_2d[1] - nose_2d[1]
        neutral_ratio = 0.65
        actual_ratio = forehead_nose_y_diff / (forehead_nose_y_diff + chin_nose_y_diff) if (forehead_nose_y_diff + chin_nose_y_diff) != 0 else 0.5
        
        if actual_ratio < neutral_ratio - 0.05:
            vertical_angle = -vertical_angle * 1.5
        elif actual_ratio > neutral_ratio + 0.05:
            vertical_angle = vertical_angle
        else:
            vertical_angle = 0
        
        # Calculate horizontal angle
        horizontal_vec = np.array([right_ear_2d[0] - left_ear_2d[0], right_ear_2d[1] - left_ear_2d[1]])
        horizontal_ref = np.array([1, 0])
        
        horizontal_vec_norm = horizontal_vec / np.linalg.norm(horizontal_vec) if np.linalg.norm(horizontal_vec) > 0 else horizontal_vec
        horizontal_ref_norm = horizontal_ref / np.linalg.norm(horizontal_ref)
        
        dot_product = np.clip(np.dot(horizontal_vec_norm, horizontal_ref_norm), -1.0, 1.0)
        horizontal_angle = np.degrees(np.arccos(dot_product))
        
        center_x = (left_ear_2d[0] + right_ear_2d[0]) / 2
        if nose_2d[0] < center_x - face_width * 0.05:
            horizontal_angle = -horizontal_angle
        elif nose_2d[0] > center_x + face_width * 0.05:
            horizontal_angle = horizontal_angle
        else:
            horizontal_angle = 0
            
        debug_info = {
            "face_height": face_height,
            "forehead_nose_y_diff": forehead_nose_y_diff,
            "chin_nose_y_diff": chin_nose_y_diff,
            "actual_ratio": actual_ratio,
            "neutral_ratio": neutral_ratio
        }
        
        return vertical_angle, horizontal_angle, debug_info

def head_posture_detection(input_source='webcam'):
    """
    Detect head posture from webcam, video file, or image file.
    
    Args:
        input_source (str): Input source - 'webcam', or path to video/image file
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
        max_num_faces=1
    )

    # Constants for head pose estimation
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]
    LEFT_EAR = 234
    RIGHT_EAR = 454
    NOSE_TIP = 4
    FOREHEAD = 10
    CHIN = 152

    # Sensitive threshold angles (in degrees) for posture detection
    TILT_UP_THRESHOLD = 1
    TILT_DOWN_THRESHOLD = -1
    TURN_LEFT_THRESHOLD = 3
    TURN_RIGHT_THRESHOLD = -3

    # For smoothing
    angle_history_size = 5
    vertical_angle_history = []
    horizontal_angle_history = []

    # Track perfect head alignment duration
    perfect_alignment_start = None
    perfect_alignment_duration = 0

    def smooth_angles(v_angle, h_angle):
        """Apply smoothing to reduce jitter in angle measurements"""
        vertical_angle_history.append(v_angle)
        horizontal_angle_history.append(h_angle)
        
        if len(vertical_angle_history) > angle_history_size:
            vertical_angle_history.pop(0)
        if len(horizontal_angle_history) > angle_history_size:
            horizontal_angle_history.pop(0)
        
        weights = np.linspace(0.5, 1.0, len(vertical_angle_history))
        weights = weights / np.sum(weights)
        
        smoothed_v_angle = np.sum(np.array(vertical_angle_history) * weights)
        smoothed_h_angle = np.sum(np.array(horizontal_angle_history) * weights)
        
        return smoothed_v_angle, smoothed_h_angle

    # Initialize video capture
    if input_source == 'webcam':
        cap = cv2.VideoCapture(0)
        is_image = False
    else:
        # Check if it's an image file
        if input_source.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(input_source)
            if image is None:
                print(f"Error: Could not read image from {input_source}")
                return
            is_image = True
        else:
            cap = cv2.VideoCapture(input_source)
            is_image = False

    if not is_image:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Processing loop
    while True:
        if is_image:
            frame = image.copy()
        else:
            success, frame = cap.read()
            if not success:
                if input_source != 'webcam':
                    print("End of video stream")
                break
        
        # Flip the image horizontally for a mirror effect (only for webcam)
        if input_source == 'webcam':
            frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Face Mesh
        results = face_mesh.process(image_rgb)
        
        # Create background info panel
        info_panel = np.zeros((180, frame.shape[1], 3), dtype=np.uint8)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get head pose angles
                vertical_angle, horizontal_angle, debug_info = calculate_head_pose(frame, face_landmarks.landmark)
                
                # Apply smoothing
                smoothed_v, smoothed_h = smooth_angles(vertical_angle, horizontal_angle)
                
                # Determine head posture
                posture = "Perfect position"
                color = (0, 255, 0)
                is_perfectly_aligned = True
                
                if smoothed_v < -TILT_UP_THRESHOLD:
                    posture = f"Head up ({abs(smoothed_v):.1f}°)"
                    color = (255, 0, 0)
                    is_perfectly_aligned = False
                elif smoothed_v > TILT_DOWN_THRESHOLD:
                    posture = f"Head down ({smoothed_v:.1f}°)"
                    color = (0, 0, 255)
                    is_perfectly_aligned = False
                
                if smoothed_h > TURN_LEFT_THRESHOLD:
                    posture = f"Head left ({smoothed_h:.1f}°)"
                    color = (255, 255, 0)
                    is_perfectly_aligned = False
                elif smoothed_h < TURN_RIGHT_THRESHOLD:
                    posture = f"Head right ({abs(smoothed_h):.1f}°)"
                    color = (0, 255, 255)
                    is_perfectly_aligned = False
                
                # Combined diagonal movement detection
                if smoothed_v < -TILT_UP_THRESHOLD and smoothed_h > TURN_LEFT_THRESHOLD:
                    posture = f"Head up-left ({abs(smoothed_v):.1f}°, {smoothed_h:.1f}°)"
                    color = (255, 0, 255)
                    is_perfectly_aligned = False
                elif smoothed_v < -TILT_UP_THRESHOLD and smoothed_h < TURN_RIGHT_THRESHOLD:
                    posture = f"Head up-right ({abs(smoothed_v):.1f}°, {abs(smoothed_h):.1f}°)"
                    color = (255, 0, 255)
                    is_perfectly_aligned = False
                elif smoothed_v > TILT_DOWN_THRESHOLD and smoothed_h > TURN_LEFT_THRESHOLD:
                    posture = f"Head down-left ({smoothed_v:.1f}°, {smoothed_h:.1f}°)"
                    color = (255, 0, 255)
                    is_perfectly_aligned = False
                elif smoothed_v > TILT_DOWN_THRESHOLD and smoothed_h < TURN_RIGHT_THRESHOLD:
                    posture = f"Head down-right ({smoothed_v:.1f}°, {abs(smoothed_h):.1f}°)"
                    color = (255, 0, 255)
                    is_perfectly_aligned = False
                
                # Track duration of perfect head alignment
                current_time = time.time()
                if is_perfectly_aligned:
                    if perfect_alignment_start is None:
                        perfect_alignment_start = current_time
                    perfect_alignment_duration = current_time - perfect_alignment_start
                else:
                    perfect_alignment_start = None
                    perfect_alignment_duration = 0
                
                # Display the posture
                cv2.putText(frame, posture, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Display precise angle information
                cv2.putText(info_panel, f"Vertical: {smoothed_v:.1f}° ({'up' if smoothed_v < 0 else 'down'})", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Horizontal: {smoothed_h:.1f}° ({'left' if smoothed_h > 0 else 'right'})", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Ratio: {debug_info['actual_ratio']:.2f} (Neutral: {debug_info['neutral_ratio']:.2f})", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if is_perfectly_aligned:
                    cv2.putText(info_panel, f"Perfect alignment: {perfect_alignment_duration:.1f}s", 
                               (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if perfect_alignment_duration > 3.0:
                        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 5)
                
                # Visualize head pose vectors
                nose_tip = (int(face_landmarks.landmark[NOSE_TIP].x * frame.shape[1]), 
                            int(face_landmarks.landmark[NOSE_TIP].y * frame.shape[0]))
                
                scale_factor = 100
                v_end = (nose_tip[0], nose_tip[1] - int(scale_factor * np.sin(np.radians(-smoothed_v))))
                h_end = (nose_tip[0] + int(scale_factor * np.sin(np.radians(smoothed_h))), nose_tip[1])
                
                cv2.line(frame, nose_tip, v_end, (0, 255, 0), 2)
                cv2.line(frame, nose_tip, h_end, (255, 0, 0), 2)
        else:
            cv2.putText(info_panel, "No face detected", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Combine main image and info panel
        combined_display = np.vstack([frame, info_panel])
        
        # Display the combined image
        cv2.imshow('Head Posture Detection', combined_display)
        
        # For single image, wait for key press
        if is_image:
            cv2.waitKey(0)
            break
        
        # Exit on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up
    if not is_image:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Head Posture Detection')
    parser.add_argument('--input', type=str, default='webcam',
                       help="Input source: 'webcam' or path to image/video file")
    
    args = parser.parse_args()
    
    head_posture_detection(args.input)