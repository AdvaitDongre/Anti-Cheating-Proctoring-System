import cv2
import dlib
import numpy as np

class PupilTracker:
    def __init__(self, shape_predictor_path):
        """
        Initialize the pupil tracker with a facial landmark predictor
        
        Args:
            shape_predictor_path: Path to dlib's facial landmark predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
    
    def detect_pupil(self, eye_region):
        """
        Detect pupil in an eye region
        
        Args:
            eye_region: Image of the eye region
            
        Returns:
            center, bbox: Pupil center coordinates and bounding box
        """
        if eye_region.size == 0:
            return None, None
            
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            pupil_contour = max(contours, key=cv2.contourArea)
            px, py, pw, ph = cv2.boundingRect(pupil_contour)
            return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
        return None, None

    def process_eye_movement(self, frame):
        """
        Process eye movement and determine gaze direction
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame, gaze_direction: Frame with annotations and detected gaze direction
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        gaze_direction = "Looking Center"
        processed_frame = frame.copy()
        
        pupil_data = {
            "left_pupil": None,
            "right_pupil": None,
            "left_eye_rect": None,
            "right_eye_rect": None
        }

        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Extract left and right eye landmarks
            left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
            
            # Get bounding rectangles for the eyes
            left_eye_rect = cv2.boundingRect(left_eye_points)
            right_eye_rect = cv2.boundingRect(right_eye_points)
            
            pupil_data["left_eye_rect"] = left_eye_rect
            pupil_data["right_eye_rect"] = right_eye_rect
            
            # Extract eye regions
            left_eye = processed_frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], 
                                left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
            right_eye = processed_frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], 
                                 right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
            
            # Detect pupils
            left_pupil, left_bbox = self.detect_pupil(left_eye)
            right_pupil, right_bbox = self.detect_pupil(right_eye)
            
            pupil_data["left_pupil"] = left_pupil
            pupil_data["right_pupil"] = right_pupil
            
            # Draw bounding boxes and pupils
            cv2.rectangle(processed_frame, (left_eye_rect[0], left_eye_rect[1]), 
                          (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
            cv2.rectangle(processed_frame, (right_eye_rect[0], right_eye_rect[1]), 
                          (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
            
            if left_pupil and left_bbox:
                cv2.circle(processed_frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 
                           5, (0, 0, 255), -1)
            if right_pupil and right_bbox:
                cv2.circle(processed_frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 
                           5, (0, 0, 255), -1)
            
            # Gaze Detection
            if left_pupil and right_pupil:
                lx, ly = left_pupil
                rx, ry = right_pupil
                
                eye_width = left_eye_rect[2]
                eye_height = left_eye_rect[3]
                norm_ly, norm_ry = ly / eye_height, ry / eye_height
                
                if lx < eye_width // 3 and rx < eye_width // 3:
                    gaze_direction = "Looking Left"
                elif lx > 2 * eye_width // 3 and rx > 2 * eye_width // 3:
                    gaze_direction = "Looking Right"
                elif norm_ly < 0.3 and norm_ry < 0.3:
                    gaze_direction = "Looking Up"
                elif norm_ly > 0.5 and norm_ry > 0.5:
                    gaze_direction = "Looking Down"
                else:
                    gaze_direction = "Looking Center"
        
        return processed_frame, gaze_direction, pupil_data