import cv2
import dlib
import numpy as np
import time
from collections import deque
import os
from scipy.spatial import distance as dist

# ========== SYSTEM CONFIGURATION ==========
CALIBRATION_TIME = 5           # 5-second calibration period
WARNING_THRESHOLD = 2          # 2 seconds looking away triggers warning
ALERT_COOLDOWN = 5             # 5 seconds between repeat warnings
STABILITY_THRESHOLD = 0.3      # Required calibration stability
MIN_FACE_HEIGHT = 150          # Minimum face size (pixels) for valid calibration

# ========== MODEL SETUP ==========
MODEL_PATH = r"C:\Users\Ritesh\OneDrive\Desktop\newniga\anti-cheating\model\shape_predictor_68_face_landmarks.dat"
if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file missing! Please check the path.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# ========== POSE DETECTION ENGINE ==========
class ProctoringSystem:
    def __init__(self):
        self.calibrated = False
        self.reference_angles = None
        self.cheating_start = None
        self.last_alert = 0
        self.current_state = "Initializing..."
        self.stability_counter = 0
        
        # 3D model points (optimized for exam scenarios)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye
            (225.0, 170.0, -135.0),  # Right eye
            (-150.0, -150.0, -125.0),# Left mouth
            (150.0, -150.0, -125.0)  # Right mouth
        ], dtype=np.float64) / 4.5
        
        # Smoothing buffers
        self.yaw_buffer = deque(maxlen=15)
        self.pitch_buffer = deque(maxlen=15)
        self.roll_buffer = deque(maxlen=15)

    def calibrate(self, frame):
        """Robust calibration routine for exam conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if not faces:
            self.current_state = "FACE NOT FOUND - Look straight at camera"
            return False
        
        # Check face size
        face_height = faces[0].bottom() - faces[0].top()
        if face_height < MIN_FACE_HEIGHT:
            self.current_state = "MOVE CLOSER TO CAMERA"
            return False
        
        landmarks = predictor(gray, faces[0])
        image_points = self._get_landmark_points(landmarks)
        angles = self._calculate_head_pose(image_points, frame.shape)
        
        # Stability check
        if self.reference_angles is not None:
            if np.allclose(angles, self.reference_angles, atol=5):
                self.stability_counter += 1
            else:
                self.stability_counter = max(0, self.stability_counter-1)
        
        self.reference_angles = angles
        return self.stability_counter >= CALIBRATION_TIME * 3  # Need stable samples
    
    def detect_cheating(self, frame):
        """Comprehensive cheating detection with visual feedback"""
        if not self.calibrated:
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if not faces:
            self._handle_no_face()
            return self._draw_ui(frame)
            
        landmarks = predictor(gray, faces[0])
        image_points = self._get_landmark_points(landmarks)
        angles = self._calculate_head_pose(image_points, frame.shape)
        
        # Smooth angles
        yaw = self._smooth_angle(self.yaw_buffer, angles[1])
        pitch = self._smooth_angle(self.pitch_buffer, angles[0])
        
        # Determine head position state
        self._update_head_state(yaw, pitch)
        
        # Cheating detection logic
        current_time = time.time()
        if self.current_state != "Looking at Screen":
            if self.cheating_start is None:
                self.cheating_start = current_time
            elif current_time - self.cheating_start > WARNING_THRESHOLD:
                if current_time - self.last_alert > ALERT_COOLDOWN:
                    self.last_alert = current_time
                    frame = self._draw_warning(frame)
        else:
            self.cheating_start = None
            
        return self._draw_ui(frame)
    
    # ========== PRIVATE METHODS ==========
    def _get_landmark_points(self, landmarks):
        return np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
        ], dtype=np.float64)
    
    def _calculate_head_pose(self, image_points, frame_shape):
        camera_matrix = np.array([
            [frame_shape[1], 0, frame_shape[1]/2],
            [0, frame_shape[1], frame_shape[0]/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        _, rotation_vector, _ = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, 
            np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
            
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return cv2.RQDecomp3x3(rotation_matrix)[0]
    
    def _smooth_angle(self, buffer, angle):
        buffer.append(angle)
        return np.mean(buffer)
    
    def _update_head_state(self, yaw, pitch):
        yaw_diff = yaw - self.reference_angles[1]
        pitch_diff = pitch - self.reference_angles[0]
        
        if abs(yaw_diff) > 15:
            self.current_state = "Looking Left" if yaw_diff < 0 else "Looking Right"
        elif abs(pitch_diff) > 10:
            self.current_state = "Looking Up" if pitch_diff > 0 else "Looking Down"
        else:
            self.current_state = "Looking at Screen"
    
    def _handle_no_face(self):
        self.current_state = "FACE NOT DETECTED"
        self.cheating_start = time.time() if self.cheating_start is None else self.cheating_start
    
    def _draw_warning(self, frame):
        cv2.rectangle(frame, (0,0), (frame.shape[1],80), (0,0,255), -1)
        cv2.putText(frame, "WARNING: POTENTIAL CHEATING", (50,50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Detected: {self.current_state}", (50,80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return frame
    
    def _draw_ui(self, frame):
        status_color = (0,255,0) if self.current_state == "Looking at Screen" else (0,0,255)
        cv2.rectangle(frame, (0,frame.shape[0]-40), (frame.shape[1],frame.shape[0]), (30,30,30), -1)
        cv2.putText(frame, f"Status: {self.current_state}", 
                   (20,frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        if not self.calibrated:
            cv2.putText(frame, "CALIBRATING - Look straight at camera", 
                       (frame.shape[1]//2-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return frame

# ========== MAIN EXECUTION ==========
def main():
    system = ProctoringSystem()
    cap = cv2.VideoCapture(0)
    
    # Calibration phase
    print("Starting calibration...")
    calibration_start = time.time()
    while time.time() - calibration_start < CALIBRATION_TIME:
        ret, frame = cap.read()
        if not ret: break
        
        system.calibrate(frame)
        frame = system.detect_cheating(frame)
        cv2.imshow("Exam Proctoring System", frame)
        if cv2.waitKey(1) == 27: break
    
    if system.stability_counter >= CALIBRATION_TIME * 3:
        system.calibrated = True
        print("Calibration successful! Monitoring for cheating...")
    else:
        print("Calibration failed! Ensure:")
        print("- Good lighting on your face")
        print("- Centered in camera view")
        print("- No objects blocking face")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Main monitoring loop
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = system.detect_cheating(frame)
        cv2.imshow("Exam Proctoring System", frame)
        
        if cv2.waitKey(1) == 27: break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()