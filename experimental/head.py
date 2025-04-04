import cv2
import numpy as np
from ultralytics import YOLO

class CheatingDetector:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self._setup_camera()
        
        # Load YOLO model
        self.model = YOLO(r"C:\Users\Ritesh\OneDrive\Desktop\newniga\anti-cheating\model\yolo12n.pt")
        
        # Guide window parameters (static area where user should stay)
        self.guide_x1, self.guide_y1 = 300, 150  # Top-left corner
        self.guide_x2, self.guide_y2 = 900, 550  # Bottom-right corner
        
        # Tracking variables
        self.cheating = False
        self.message = ""
        self.frame_count = 0

    def _setup_camera(self):
        """Basic camera setup with color"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)

    def _detect_face(self, frame):
        """Detect faces and return bounding box"""
        results = self.model(frame, verbose=False)
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            return True, box.astype(int)
        return False, None

    def _check_if_inside_guide(self, face_box):
        """Check if face is within guide window"""
        fx1, fy1, fx2, fy2 = face_box
        
        # Check if any corner of face box is outside guide area
        if (fx1 < self.guide_x1 or fx2 > self.guide_x2 or 
            fy1 < self.guide_y1 or fy2 > self.guide_y2):
            
            # Determine direction of violation
            self.message = ""
            if fx1 < self.guide_x1: self.message += "MOVED LEFT "
            if fx2 > self.guide_x2: self.message += "MOVED RIGHT "
            if fy1 < self.guide_y1: self.message += "MOVED UP "
            if fy2 > self.guide_y2: self.message += "MOVED DOWN"
            return False
        return True

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, False

        # Draw static guide window (semi-transparent green)
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (self.guide_x1, self.guide_y1),
                     (self.guide_x2, self.guide_y2),
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        cv2.rectangle(frame, 
                     (self.guide_x1, self.guide_y1),
                     (self.guide_x2, self.guide_y2),
                     (0, 255, 0), 2)

        # Detect face every 5 frames
        if self.frame_count % 5 == 0:
            detected, face_box = self._detect_face(frame)
            
            if detected:
                # Draw face box
                cv2.rectangle(frame, 
                             (face_box[0], face_box[1]),
                             (face_box[2], face_box[3]),
                             (0, 165, 255), 2)
                
                # Check position
                self.cheating = not self._check_if_inside_guide(face_box)
            else:
                self.cheating = True
                self.message = "FACE NOT DETECTED"

        self.frame_count += 1

        # Display warning if outside guide area
        if self.cheating:
            cv2.putText(frame, f"WARNING: {self.message}", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), 
                        (frame.shape[1], frame.shape[0]), 
                        (0, 0, 255), 10)

        return frame, True

    def release(self):
        self.cap.release()

def main():
    detector = CheatingDetector()
    
    try:
        while True:
            frame, success = detector.process_frame()
            if not success:
                break
                
            cv2.imshow("Exam Proctoring - Stay Inside Green Box", frame)
            
            if cv2.waitKey(1) == 27:  # ESC to exit
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()