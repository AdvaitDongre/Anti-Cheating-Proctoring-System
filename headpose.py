import cv2
import numpy as np
import dlib
from ultralytics import YOLO
from collections import deque

class ExamProctor:
    def __init__(self, yolo_path, shape_predictor_path):
        """Initialize proctor with model paths"""
        self.yolo = YOLO(yolo_path)
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()
        
        # **Optimized thresholds for more accurate detection**
        self.thresholds = {
            'looking_up': 10,         # Increased tolerance
            'looking_down': -10,
            'looking_left': -18,      # Increased tolerance for left/right
            'looking_right': 18,
            'max_tilt': 8             # Slight tilt allowed
        }
        
        # Anti-cheating tracking
        self.look_away_counter = 0
        self.max_look_away_frames = 15  # ~0.5 sec before flagging cheating
        self.attention_history = deque(maxlen=30)

        # Landmark indices
        self.LANDMARKS = {
            'nose_tip': 30, 'chin': 8,
            'left_eye': 36, 'right_eye': 45,
            'left_ear': 0, 'right_ear': 16
        }

    def get_landmarks(self, gray_frame, face_rect):
        """Detects facial landmarks"""
        landmarks = self.shape_predictor(gray_frame, face_rect)
        return np.array([[p.x, p.y] for p in landmarks.parts()])

    def calculate_head_pose(self, frame, landmarks):
        """Calculates head angles (Pitch, Yaw, Roll)"""
        nose = landmarks[self.LANDMARKS['nose_tip']]
        chin = landmarks[self.LANDMARKS['chin']]
        left_eye = landmarks[self.LANDMARKS['left_eye']]
        right_eye = landmarks[self.LANDMARKS['right_eye']]
        left_ear = landmarks[self.LANDMARKS['left_ear']]
        right_ear = landmarks[self.LANDMARKS['right_ear']]
        
        # 3D reference points
        model_points = np.array([
            (0.0, 0.0, 0.0),          
            (0.0, -330.0, -65.0),    
            (-225.0, 170.0, -135.0),  
            (225.0, 170.0, -135.0),   
            (-500.0, 0.0, -250.0),    
            (500.0, 0.0, -250.0)      
        ])
        
        # Camera calibration
        size = frame.shape
        focal_length = size[1]
        camera_matrix = np.array([
            [focal_length, 0, size[1] / 2],
            [0, focal_length, size[0] / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Solve PnP
        image_points = np.array([nose, chin, left_eye, right_eye, left_ear, right_ear], dtype=np.float32)
        _, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, None)
        
        # Convert to angles
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1], angles[2]  # Pitch, Yaw, Roll

    def determine_position(self, angles):
        """Determines head position based on angles"""
        pitch, yaw, roll = angles
        position = "LOOKING AT SCREEN"

        if pitch > self.thresholds['looking_up']:
            position = "LOOKING UP"
        elif pitch < self.thresholds['looking_down']:
            position = "LOOKING DOWN"
        
        if yaw < self.thresholds['looking_left']:
            position = "LOOKING LEFT"
        elif yaw > self.thresholds['looking_right']:
            position = "LOOKING RIGHT"
        
        if abs(roll) > self.thresholds['max_tilt']:
            position += f" | TILTED {'LEFT' if roll < 0 else 'RIGHT'}"

        return position

    def process_frame(self, frame):
        """Processes each frame for cheating detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        annotated = frame.copy()
        cheating = False
        position = "NO FACE DETECTED"
        
        for face in faces:
            try:
                # Landmark detection
                landmarks = self.get_landmarks(gray, face)
                angles = self.calculate_head_pose(frame, landmarks)
                position = self.determine_position(angles)

                # Draw face bounding box
                color = (0, 255, 0)  # Green (normal)
                if position != "LOOKING AT SCREEN":
                    self.look_away_counter += 1
                else:
                    self.look_away_counter = max(0, self.look_away_counter - 2)

                if self.look_away_counter >= self.max_look_away_frames:
                    cheating = True
                    color = (0, 0, 255)  # Red (cheating detected)

                cv2.rectangle(annotated, 
                    (face.left(), face.top()), 
                    (face.right(), face.bottom()), 
                    color, 3)

                # Display text
                cv2.putText(annotated, position,
                    (face.left(), face.top() - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

            except Exception as e:
                print(f"Landmark error: {e}")

        # If the face is missing, count as looking away
        if len(faces) == 0:
            self.look_away_counter += 1
            if self.look_away_counter >= self.max_look_away_frames:
                cheating = True
                cv2.putText(annotated, "NO FACE DETECTED", 
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255), 2)

        # Show alert if cheating detected
        if cheating:
            cv2.putText(annotated, "CHEATING ALERT!", 
                (annotated.shape[1]//2-100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

        return annotated, cheating, position

    def run(self):
        """Starts live proctoring"""
        cap = cv2.VideoCapture(0)
        print("Proctoring active. Press Q to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame, cheating, position = self.process_frame(frame)

            # Show the proctoring window
            cv2.imshow("Exam Proctor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Run the proctor
if __name__ == "__main__":
    proctor = ExamProctor(
        yolo_path=r'C:\Users\Ritesh\OneDrive\Desktop\newniga\anti-cheating\model\yolo12n.pt',
        shape_predictor_path=r'C:\Users\Ritesh\OneDrive\Desktop\newniga\anti-cheating\model\shape_predictor_68_face_landmarks.dat'
    )
    proctor.run()
