import cv2
import numpy as np
import dlib
from ultralytics import YOLO
from collections import deque

class ExamProctor:
    def __init__(self, yolo_path, shape_predictor_path):
        """Initialize proctor with your exact model paths"""
        # Load models (keep your exact paths)
        self.yolo = YOLO(yolo_path)
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()
        
        # PRESERVE THESE OPTIMAL THRESHOLDS (degrees)
        self.thresholds = {
            'looking_up': 8,        # Perfect for detecting upward glances
            'looking_down': -8,     # Ideal for paper-cheating detection
            'looking_left': -15,    # Catches side-glances perfectly
            'looking_right': 15,
            'max_tilt': 5           # Detects subtle head tilts
        }
        
        # Tracking setup (DON'T MODIFY)
        self.look_away_counter = 0
        self.max_look_away_frames = 10  # 10 frames buffer
        self.attention_history = deque(maxlen=30)
        
        # Landmark indices (68-point model)
        self.LANDMARKS = {
            'nose_tip': 30, 'chin': 8,
            'left_eye': 36, 'right_eye': 45,
            'left_eyebrow': 17, 'right_eyebrow': 26
        }

    def get_landmarks(self, gray_frame, face_rect):
        """EXACT facial landmark detection (preserve this)"""
        landmarks = self.shape_predictor(gray_frame, face_rect)
        return np.array([[p.x, p.y] for p in landmarks.parts()])

    def calculate_head_pose(self, frame, landmarks):
        """PRECISE angle calculation (core logic - don't change)"""
        nose = landmarks[self.LANDMARKS['nose_tip']]
        chin = landmarks[self.LANDMARKS['chin']]
        left_eye = landmarks[self.LANDMARKS['left_eye']]
        right_eye = landmarks[self.LANDMARKS['right_eye']]
        
        # 3D reference points
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-225.0, 170.0, -135.0), # Left eye
            (225.0, 170.0, -135.0)   # Right eye
        ])
        
        # Camera calibration
        size = frame.shape
        focal_length = size[1]
        camera_matrix = np.array(
            [[focal_length, 0, size[1]/2],
             [0, focal_length, size[0]/2],
             [0, 0, 1]], dtype=np.float32
        )
        
        # Solve PnP
        image_points = np.array([nose, chin, left_eye, right_eye], dtype=np.float32)
        _, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, None)
        
        # Convert to angles
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1], angles[2]  # Pitch, Yaw, Roll

    def determine_position(self, angles):
        """HEAD POSITION LOGIC (keep this exact)"""
        pitch, yaw, roll = angles
        
        # Primary position
        if pitch > self.thresholds['looking_up']:
            position = "LOOKING UP"
        elif pitch < self.thresholds['looking_down']:
            position = "LOOKING DOWN"
        else:
            position = "LOOKING AT SCREEN"
        
        # Secondary direction
        if yaw < self.thresholds['looking_left']:
            position += " | LEFT"
        elif yaw > self.thresholds['looking_right']:
            position += " | RIGHT"
        
        # Tilt detection
        if abs(roll) > self.thresholds['max_tilt']:
            position += f" | TILTED {'LEFT' if roll < 0 else 'RIGHT'}"
        
        return position

    def process_frame(self, frame):
        """CORE PROCESSING PIPELINE (preserve this structure)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        annotated = frame.copy()
        cheating = False
        
        for face in faces:
            try:
                # Face bounding box (thick green when normal)
                cv2.rectangle(annotated, 
                    (face.left(), face.top()),
                    (face.right(), face.bottom()),
                    (0, 255, 0), 3)
                
                # Landmark detection
                landmarks = self.get_landmarks(gray, face)
                angles = self.calculate_head_pose(frame, landmarks)
                position = self.determine_position(angles)
                
                # Cheating detection
                if ("UP" in position or "DOWN" in position or 
                    "LEFT" in position or "RIGHT" in position):
                    self.look_away_counter += 1
                    if self.look_away_counter >= self.max_look_away_frames:
                        cheating = True
                        # Red alert box
                        cv2.rectangle(annotated, 
                            (face.left()-10, face.top()-10),
                            (face.right()+10, face.bottom()+10),
                            (0, 0, 255), 3)
                else:
                    self.look_away_counter = max(0, self.look_away_counter-2)
                
                # Display info
                cv2.putText(annotated, position,
                    (face.left(), face.top() - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
                
                # Landmark dots (optional)
                for (x, y) in landmarks:
                    cv2.circle(annotated, (x, y), 1, (0, 0, 255), -1)
                    
            except Exception as e:
                print(f"Landmark error: {e}")
        
        # No face detection
        if len(faces) == 0:
            self.look_away_counter += 1
            if self.look_away_counter >= self.max_look_away_frames:
                cheating = True
                cv2.putText(annotated, "NO FACE DETECTED", 
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255), 2)
        
        return annotated, cheating

    def run(self):
        """MAIN LOOP (keep this execution flow)"""
        cap = cv2.VideoCapture(0)
        print("Proctoring active. Press Q to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame, cheating = self.process_frame(frame)
            
            # Display alert if cheating
            if cheating:
                cv2.putText(frame, "CHEATING ALERT!", 
                    (frame.shape[1]//2-100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)
            
            # Show frame
            cv2.imshow("Exam Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Initialize with your exact paths
if __name__ == "__main__":
    proctor = ExamProctor(
        yolo_path=r'C:\Users\Ritesh\OneDrive\Desktop\newniga\anti-cheating\model\yolo12n.pt',
        shape_predictor_path=r'C:\Users\Ritesh\OneDrive\Desktop\newniga\anti-cheating\model\shape_predictor_68_face_landmarks.dat'
    )
    proctor.run()