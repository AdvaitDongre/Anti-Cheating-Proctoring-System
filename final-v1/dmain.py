import cv2
import json
import time
from datetime import datetime
from collections import defaultdict
import threading
import numpy as np
from detect_blinks import blink_detection
from gaze_detector import gaze_detection
from head_posture import head_posture_detection
from lip_movement import LipMovementDetector
from tile import check_user_position, detector as tile_detector
from yolo_detecting_multiple_classes import YOLOv12ExamCheatingDetector
from process_monitor import AdvancedProcessMonitor
from speach_detector import GeminiSpeakerDetector
from sitelocker import FullSiteLocker
import dlib
import os

class ExamMonitoringSystem:
    def __init__(self):
        # Initialize all detectors
        self.blink_data = []
        self.gaze_data = []
        self.head_posture_data = []
        self.lip_movement_data = []
        self.tile_violations = 0
        self.yolo_detections = []
        self.process_monitor_data = []
        self.speaker_detections = []
        self.start_time = time.time()
        self.running = True
        
        # Initialize detectors
        self.blink_detector = blink_detection
        self.gaze_detector = gaze_detection
        self.head_posture_detector = head_posture_detection
        self.lip_detector = LipMovementDetector()
        
        # Initialize YOLO detector with proper model path
        model_path = '../model/yolo12n.pt'
        if os.path.exists(model_path):
            self.yolo_detector = YOLOv12ExamCheatingDetector(model_path=model_path)
        else:
            print(f"Warning: YOLO model not found at {model_path}. YOLO detection disabled.")
            self.yolo_detector = None
        
        # Initialize process monitor
        self.process_monitor = AdvancedProcessMonitor()
        
        # Initialize speaker detector and site locker
        self.speaker_detector = None  # Will be initialized if needed
        self.site_locker = None  # Will be initialized if needed
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        # For tile detection
        self.tile_detector = dlib.get_frontal_face_detector()
        
        # Start monitoring threads
        self.start_monitoring_threads()
        
    def start_monitoring_threads(self):
        # Start process monitoring in a separate thread
        self.process_thread = threading.Thread(target=self.monitor_processes)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start speaker detection if needed
        if os.getenv("GEMINI_API_KEY"):
            self.speaker_detector = GeminiSpeakerDetector(api_key=os.getenv("GEMINI_API_KEY"))
            self.speaker_thread = threading.Thread(target=self.monitor_speakers)
            self.speaker_thread.daemon = True
            self.speaker_thread.start()
        
    def monitor_processes(self):
        while self.running:
            suspicious = self.process_monitor.scan_processes()
            if suspicious:
                timestamp = time.time() - self.start_time
                self.process_monitor_data.append({
                    "timestamp": timestamp,
                    "processes": suspicious
                })
            time.sleep(5)  # Check every 5 seconds
            
    def monitor_speakers(self):
        if self.speaker_detector:
            self.speaker_detector.start()
            
    def process_frame(self, frame):
        timestamp = time.time() - self.start_time
        
        # Blink detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dlib.get_frontal_face_detector()(gray)
        
        if faces:
            # Get the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            shape = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Blink detection
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            left_ear = (np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / (2 * np.linalg.norm(left_eye[0] - left_eye[3]))
            right_ear = (np.linalg.norm(right_eye[1] - right_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[4])) / (2 * np.linalg.norm(right_eye[0] - right_eye[3]))
            avg_ear = (left_ear + right_ear) / 2
            is_blinking = avg_ear < 0.45
            self.blink_data.append({
                "timestamp": timestamp,
                "is_blinking": is_blinking,
                "ear": avg_ear
            })
            
            # Gaze detection
            gaze_direction = "center"
            try:
                # Simplified gaze detection
                left_eye_center = np.mean(left_eye, axis=0)
                right_eye_center = np.mean(right_eye, axis=0)
                
                # Determine gaze direction based on eye positions
                if left_eye_center[0] < right_eye_center[0] - 10:
                    gaze_direction = "left"
                elif left_eye_center[0] > right_eye_center[0] + 10:
                    gaze_direction = "right"
                elif left_eye_center[1] < right_eye_center[1] - 5:
                    gaze_direction = "up"
                elif left_eye_center[1] > right_eye_center[1] + 5:
                    gaze_direction = "down"
            except:
                pass
                
            self.gaze_data.append({
                "timestamp": timestamp,
                "direction": gaze_direction
            })
            
            # Head posture detection
            vertical_angle = 0
            horizontal_angle = 0
            try:
                # Simplified head posture detection
                nose_tip = shape[30]
                chin = shape[8]
                left_side = shape[0]
                right_side = shape[16]
                
                # Calculate vertical angle
                vertical_vec = nose_tip - chin
                vertical_angle = np.degrees(np.arctan2(vertical_vec[1], vertical_vec[0])) - 90
                
                # Calculate horizontal angle
                horizontal_vec = right_side - left_side
                horizontal_angle = np.degrees(np.arctan2(horizontal_vec[1], horizontal_vec[0]))
            except:
                pass
                
            self.head_posture_data.append({
                "timestamp": timestamp,
                "vertical_angle": vertical_angle,
                "horizontal_angle": horizontal_angle
            })
            
            # Lip movement detection
            lip_movement, mouth_opened, openness = self.lip_detector._detect_movement(0)  # Simplified
            self.lip_movement_data.append({
                "timestamp": timestamp,
                "is_moving": lip_movement,
                "mouth_opened": mouth_opened,
                "openness": openness
            })
            
            # Tile detection
            padding = 10
            x, y = face.left() - padding, face.top() - padding
            w, h = face.width() + 2*padding, face.height() + 2*padding
            
            # Check if face is outside the tile (150, 80, 500, 420)
            is_outside = (x < 150 or y < 80 or (x + w) > 500 or (y + h) > 420)
            if is_outside:
                self.tile_violations += 1
                
        # YOLO object detection
        try:
            if self.yolo_detector:
                results = self.yolo_detector.model.predict(
                    source=frame,
                    conf=0.5,
                    iou=0.5,
                    verbose=False
                )
                
                detections = []
                for detection in results[0].boxes:
                    cls = int(detection.cls)
                    conf = float(detection.conf)
                    box = detection.xyxy[0].tolist()
                    detections.append({
                        "class_id": cls,
                        "confidence": conf,
                        "box": box
                    })
                    
                self.yolo_detections.append({
                    "timestamp": timestamp,
                    "detections": detections
                })
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            # Add empty detection for this timestamp to maintain data consistency
            self.yolo_detections.append({
                "timestamp": timestamp,
                "detections": []
            })
            
    def run(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process frame in the background
                self.process_frame(frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
            
    def cleanup(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.speaker_detector:
            self.speaker_detector.stop()
            
        # Save all data to JSON
        self.save_results()
        
    def save_results(self):
        results = {
            "metadata": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "duration": time.time() - self.start_time,
                "system": "Exam Monitoring System"
            },
            "blink_data": self.blink_data,
            "gaze_data": self.gaze_data,
            "head_posture_data": self.head_posture_data,
            "lip_movement_data": self.lip_movement_data,
            "tile_violations": self.tile_violations,
            "yolo_detections": self.yolo_detections,
            "process_monitor_data": self.process_monitor_data,
            "speaker_detections": self.speaker_detections if hasattr(self, 'speaker_detections') else []
        }
        
        with open("exam_monitoring_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("Results saved to exam_monitoring_results.json")

if __name__ == "__main__":
    monitor = ExamMonitoringSystem()
    monitor.run()