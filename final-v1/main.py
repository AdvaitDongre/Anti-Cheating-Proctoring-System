import cv2
import threading
import time
import numpy as np
import os
from dotenv import load_dotenv
from collections import deque

# Import all detection modules
from detect_blinks import blink_detection
from gaze_detector import process_frame as process_gaze_frame
from lip_movement import LipMovementDetector
from process_monitor import AdvancedProcessMonitor
from yolo_detecting_multiple_classes import YOLOv12ExamCheatingDetector
from sitelocker import FullSiteLocker
from speach_detector import GeminiSpeakerDetector

import imutils
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

class CheatingDetectionSystem:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self.processed_frames = {}
        self.detection_results = {}
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load environment variables for API keys if needed
        load_dotenv()
        
        # Initialize detectors
        self.init_detectors()
        
        # For UI display
        self.total_blinks = 0
        self.blink_state = False
        self.gaze_direction = "center"
        self.mouth_status = {"moving": False, "opened": False, "openness": 0}
        self.suspicious_processes = []
        self.yolo_detections = {}
        
        # For display grid
        self.display_width = 1280
        self.display_height = 720
        self.grid_layout = (2, 2)  # rows, cols
        
    def init_detectors(self):
        """Initialize all detection modules"""
        # Lip movement detector
        self.lip_detector = LipMovementDetector()
        
        # Process monitor
        self.process_monitor = AdvancedProcessMonitor()
        
        # YOLO detector (if model file is available)
        model_path = '../model/yolo12n.pt'
        if os.path.exists(model_path):
            self.yolo_detector = YOLOv12ExamCheatingDetector(model_path=model_path)
        else:
            print(f"Warning: YOLO model not found at {model_path}. YOLO detection disabled.")
            self.yolo_detector = None
        
        # For speech detection, check if Gemini API key is available
        if "GEMINI_API_KEY" in os.environ:
            self.speech_detector = GeminiSpeakerDetector(api_key=os.environ["GEMINI_API_KEY"])
        else:
            print("Warning: GEMINI_API_KEY not found. Speech detection disabled.")
            self.speech_detector = None
            
        # Site locker (initialized but not started by default)
        self.site_locker = FullSiteLocker(allowed_url="www.google.com", test_duration=3600)
        
    def start(self):
        """Start all detection threads"""
        if self.running:
            return
            
        self.running = True
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start detection threads
        self.start_detection_threads()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self.display_results)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        # Start process monitoring in a separate thread
        self.process_thread = threading.Thread(target=self.monitor_processes)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Optionally start speech detection
        if self.speech_detector:
            self.speech_detector.start()
    
    def start_detection_threads(self):
        """Start all detection threads"""
        # Blink detection thread
        self.blink_thread = threading.Thread(target=self.run_blink_detection)
        self.blink_thread.daemon = True
        self.blink_thread.start()
        
        # Gaze detection thread
        self.gaze_thread = threading.Thread(target=self.run_gaze_detection)
        self.gaze_thread.daemon = True
        self.gaze_thread.start()
        
        # Lip movement detection thread
        self.lip_thread = threading.Thread(target=self.run_lip_detection)
        self.lip_thread.daemon = True
        self.lip_thread.start()
        
        # YOLO detection thread (if available)
        if self.yolo_detector:
            self.yolo_thread = threading.Thread(target=self.run_yolo_detection)
            self.yolo_thread.daemon = True
            self.yolo_thread.start()
    
    def capture_frames(self):
        """Continuously capture frames from webcam"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                time.sleep(0.1)
                continue
                
            # Flip frame horizontally for more intuitive display
            frame = cv2.flip(frame, 1)
            
            with self.lock:
                self.frame = frame.copy()
            
            # Slight delay to avoid excessive CPU usage
            time.sleep(0.01)
    
    def run_blink_detection(self):
        """Run blink detection continuously using detect_blinks.py functionality"""
        # Setup from detect_blinks.py
        def calculate_EAR(eye):
            y1 = dist.euclidean(eye[1], eye[5])
            y2 = dist.euclidean(eye[2], eye[4])
            x1 = dist.euclidean(eye[0], eye[3])
            EAR = (y1+y2) / x1
            return EAR
            
        blink_thresh = 0.45
        succ_frame = 2
        count_frame = 0
        
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        
        detector = dlib.get_frontal_face_detector()
        landmark_predict = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
        
        while self.running:
            # Get the latest frame
            with self.lock:
                if self.frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.frame.copy()
            
            frame_displayed = frame.copy()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = detector(img_gray)
            for face in faces:
                shape = landmark_predict(img_gray, face)
                shape = face_utils.shape_to_np(shape)
                
                lefteye = shape[L_start: L_end]
                righteye = shape[R_start:R_end]
                
                # Use the calculate_EAR function defined locally
                left_EAR = calculate_EAR(lefteye)
                right_EAR = calculate_EAR(righteye)
                
                avg = (left_EAR+right_EAR)/2
                
                # Draw landmarks for better visualization
                for (x, y) in shape:
                    cv2.circle(frame_displayed, (x, y), 2, (0, 255, 0), -1)
                
                # Draw eye landmarks with hulls
                for eye in [lefteye, righteye]:
                    eye_hull = cv2.convexHull(eye)
                    cv2.drawContours(frame_displayed, [eye_hull], -1, (0, 255, 0), 1)
                
                if avg < blink_thresh:
                    count_frame += 1
                    self.blink_state = True
                else:
                    if count_frame >= succ_frame:
                        self.total_blinks += 1
                    count_frame = 0
                    self.blink_state = False
                
                # Draw face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(frame_displayed, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add text for blink information
            cv2.putText(frame_displayed, f'Blinks: {self.total_blinks}', (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            if self.blink_state:
                cv2.putText(frame_displayed, 'Blink Detected', (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 0), 2)
            
            # Display EAR value
            if faces:
                cv2.putText(frame_displayed, f'EAR: {avg:.2f}', (10, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
            
            # Update processed frame for display
            with self.lock:
                self.processed_frames['blink'] = frame_displayed
                self.detection_results['blink'] = {'blinks': self.total_blinks, 'blinking': self.blink_state}
            
            time.sleep(0.03)  # ~30fps processing
    
    def run_gaze_detection(self):
        """Run gaze detection continuously"""
        while self.running:
            # Get the latest frame
            with self.lock:
                if self.frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.frame.copy()
            
            # Process frame for gaze detection
            gaze_frame = process_gaze_frame(frame)
            
            # Extract gaze direction from the frame
            direction_text = "center"  # Default
            if "Direction: left" in str(gaze_frame):
                direction_text = "left"
            elif "Direction: right" in str(gaze_frame):
                direction_text = "right"
            elif "Direction: up" in str(gaze_frame):
                direction_text = "up"
            elif "Direction: down" in str(gaze_frame):
                direction_text = "down"
            
            self.gaze_direction = direction_text
            
            # Update processed frame for display
            with self.lock:
                self.processed_frames['gaze'] = gaze_frame
                self.detection_results['gaze'] = {'direction': self.gaze_direction}
            
            time.sleep(0.03)  # ~30fps processing
    
    def run_lip_detection(self):
        """Run lip movement detection continuously"""
        while self.running:
            # Get the latest frame
            with self.lock:
                if self.frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.frame.copy()
            
            # Process frame for lip movement detection
            moving, opened, openness, lip_frame = self.lip_detector.process_frame(frame)
            
            # Update mouth status
            self.mouth_status = {
                "moving": moving,
                "opened": opened,
                "openness": openness
            }
            
            # Update processed frame for display
            with self.lock:
                self.processed_frames['lip'] = lip_frame
                self.detection_results['lip'] = self.mouth_status
            
            time.sleep(0.03)  # ~30fps processing
    
    def run_yolo_detection(self):
        """Run YOLO detection for prohibited objects"""
        while self.running:
            # Get the latest frame
            with self.lock:
                if self.frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.frame.copy()
            
            # Process frame for YOLO detection
            yolo_frame, detections = self.yolo_detector.detect(frame)
            
            # Update detections
            self.yolo_detections = detections
            
            # Update processed frame for display
            with self.lock:
                self.processed_frames['yolo'] = yolo_frame
                self.detection_results['yolo'] = self.yolo_detections
            
            time.sleep(0.1)  # YOLO is more computationally intensive, run at lower fps
    
    def monitor_processes(self):
        """Monitor processes for suspicious activity"""
        while self.running:
            suspicious = self.process_monitor.scan_processes()
            if suspicious:
                self.process_monitor.log_findings(suspicious)
                self.suspicious_processes = suspicious
            
            # Wait 10 seconds between scans to save resources
            time.sleep(10)
    
    def create_dashboard(self):
        """Create a consolidated dashboard view of all detections"""
        # Start with a blank canvas
        dashboard = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Get available processed frames
        available_frames = list(self.processed_frames.keys())
        
        # Calculate grid cell dimensions
        cell_width = self.display_width // self.grid_layout[1]
        cell_height = self.display_height // self.grid_layout[0]
        
        # Place frames in grid
        for idx, key in enumerate(available_frames):
            if idx >= self.grid_layout[0] * self.grid_layout[1]:
                break  # Don't exceed grid capacity
                
            row = idx // self.grid_layout[1]
            col = idx % self.grid_layout[1]
            
            frame = self.processed_frames[key]
            # Resize the frame to fit the grid cell
            resized_frame = cv2.resize(frame, (cell_width, cell_height))
            
            # Place in dashboard
            y_start = row * cell_height
            y_end = (row + 1) * cell_height
            x_start = col * cell_width
            x_end = (col + 1) * cell_width
            
            dashboard[y_start:y_end, x_start:x_end] = resized_frame
            
            # Add label for each display
            label_text = key.upper()
            cv2.putText(dashboard, label_text, (x_start + 5, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add status summary at the bottom
        y_pos = self.display_height - 80
        cv2.putText(dashboard, f"Blinks: {self.total_blinks} | Gaze: {self.gaze_direction} | Lips: {'Moving' if self.mouth_status['moving'] else 'Still'}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        # Add YOLO detections if available
        if 'yolo' in self.detection_results:
            detections = []
            for obj, count in self.detection_results['yolo'].items():
                if count > 0:
                    detections.append(f"{obj}: {count}")
            
            if detections:
                cv2.putText(dashboard, "Detected: " + ", ".join(detections), 
                            (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return dashboard
    
    def display_results(self):
        """Display all detection results in a combined window"""
        while self.running:
            with self.lock:
                if not self.processed_frames:
                    time.sleep(0.1)
                    continue
                
                # Create dashboard display
                dashboard = self.create_dashboard()
                
                # Show the dashboard
                cv2.imshow("Anti-Cheating System Dashboard", dashboard)
                
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
    
    def stop(self):
        """Stop all threads and release resources"""
        self.running = False
        
        # Stop speech detector if running
        if self.speech_detector and hasattr(self.speech_detector, 'stop'):
            self.speech_detector.stop()
        
        # Wait for threads to complete
        time.sleep(1)
        
        # Release webcam
        self.cap.release()
        
        # Close all windows
        cv2.destroyAllWindows()
        print("Anti-cheating system stopped")

if __name__ == "__main__":
    print("Starting Anti-Cheating Detection System...")
    system = CheatingDetectionSystem()
    
    try:
        system.start()
        # Keep main thread alive with a simple loop
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop() 