import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class LipMovementDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        
        # Lip landmarks (inner and outer)
        self.upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 308]
        self.all_lip_indices = list(set(self.upper_lip_indices + self.lower_lip_indices))
        
        # Detection parameters
        self.history_length = 5
        self.movement_threshold = 0.015
        self.min_openness_change = 0.03
        self.openness_threshold = 0.09  # Threshold for mouth fully opened
        self.required_movement_frames = 2
        
        # Tracking variables
        self.openness_history = deque(maxlen=self.history_length)
        self.movement_counter = 0
        self.mouth_opened = False
    
    def _get_normalized_lip_distance(self, landmarks, frame_shape):
        """Calculate normalized mouth openness"""
        frame_height, frame_width = frame_shape[:2]
        
        # Get upper lip points
        upper_lip_points = []
        for idx in self.upper_lip_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            upper_lip_points.append([x, y])
        upper_center = np.mean(upper_lip_points, axis=0)
        
        # Get lower lip points
        lower_lip_points = []
        for idx in self.lower_lip_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            lower_lip_points.append([x, y])
        lower_center = np.mean(lower_lip_points, axis=0)
        
        # Calculate vertical distance
        lip_distance = np.linalg.norm(upper_center - lower_center)
        
        # Normalize by face width
        face_width = abs(landmarks.landmark[454].x - landmarks.landmark[234].x) * frame_width
        if face_width < 1:
            return 0
        
        return lip_distance / face_width
    
    def _detect_movement(self, current_openness):
        """Detect lip movement and fully opened state"""
        # Check if mouth is fully opened
        self.mouth_opened = current_openness >= self.openness_threshold
        
        # Movement detection logic
        if len(self.openness_history) < 1:
            self.openness_history.append(current_openness)
            return False
        
        last_openness = self.openness_history[-1]
        change = abs(current_openness - last_openness)
        self.openness_history.append(current_openness)
        
        if change > self.min_openness_change:
            self.movement_counter += 1
        else:
            self.movement_counter = max(0, self.movement_counter - 1)
        
        return self.movement_counter >= self.required_movement_frames
    
    def process_frame(self, frame):
        """Process each frame for detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        movement_detected = False
        openness = 0
        annotated_frame = frame.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                openness = self._get_normalized_lip_distance(face_landmarks, frame.shape)
                movement_detected = self._detect_movement(openness)
                
                # Draw landmarks
                for idx in self.all_lip_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)
                
                # Determine status text and color
                if self.mouth_opened:
                    status_text = "MOUTH OPEN"
                    color = (0, 255, 255)  # Yellow for opened state
                elif movement_detected:
                    status_text = "MOVING"
                    color = (0, 255, 0)    # Green for movement
                else:
                    status_text = "still"
                    color = (0, 0, 255)    # Red for still
                
                # Display status and metrics
                cv2.putText(annotated_frame, f"Status: {status_text}", (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(annotated_frame, f"Openness: {openness:.3f}", (20, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Threshold: {self.openness_threshold}", (20, 140),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                
                # Draw threshold indicator
                threshold_pos = int(self.openness_threshold * 100)
                cv2.line(annotated_frame, (300, 50), (300 + threshold_pos, 50), (0, 255, 255), 3)
        
        return movement_detected, self.mouth_opened, openness, annotated_frame

def main():
    detector = LipMovementDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        movement_detected, mouth_opened, openness, processed_frame = detector.process_frame(frame)
        
        # Debug output
        print(f"Openness: {openness:.4f} | Moving: {movement_detected} | Opened: {mouth_opened}")
        
        cv2.imshow('Lip Movement Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()