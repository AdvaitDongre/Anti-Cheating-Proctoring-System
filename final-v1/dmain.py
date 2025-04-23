import cv2
import numpy as np
import threading
import time
import argparse
from queue import Queue
from collections import deque

# Import all our detection modules
from detect_blinks import blink_detection
from gaze_detector import gaze_detection
from lip_movement import LipMovementDetector
from yolo_detecting_multiple_classes import YOLOv12ExamCheatingDetector
from process_monitor import AdvancedProcessMonitor
from speach_detector import GeminiSpeakerDetector
from sitelocker import FullSiteLocker

class CheatingDetectionSystem:
    def __init__(self, input_source='webcam', show_all=True, enable_site_lock=False, allowed_url=None):
        """
        Initialize the cheating detection system with all modules.
        
        Args:
            input_source (str): 'webcam' or path to video/image file
            show_all (bool): Whether to show all detection outputs
            enable_site_lock (bool): Whether to enable website locking
            allowed_url (str): URL to lock to if site locking is enabled
        """
        self.input_source = input_source
        self.show_all = show_all
        self.running = False
        
        # Initialize detection modules
        self.lip_detector = LipMovementDetector()
        self.yolo_detector = YOLOv12ExamCheatingDetector()
        self.process_monitor = AdvancedProcessMonitor()
        self.speaker_detector = GeminiSpeakerDetector(api_key="your-api-key") if hasattr(self, 'GeminiSpeakerDetector') else None
        
        # Site locker if enabled
        self.site_locker = None
        if enable_site_lock and allowed_url:
            self.site_locker = FullSiteLocker(allowed_url=allowed_url, test_duration=3600)  # 1 hour duration
            
        # Queues for inter-thread communication
        self.frame_queue = Queue(maxsize=1)
        self.output_frame_queue = Queue(maxsize=1)
        
        # Detection states
        self.detection_states = {
            'blink': {'count': 0, 'state': 'Open'},
            'gaze': {'direction': 'center', 'looking_at_screen': True},
            'lips': {'movement': False, 'open': False, 'openness': 0},
            'objects': {'counts': {}, 'warnings': []},
            'audio': {'speakers': 1, 'alert': False}
        }
        
        # Display parameters
        self.display_width = 1280
        self.display_height = 720
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.text_color = (255, 255, 255)
        self.alert_color = (0, 0, 255)
        
    def start(self):
        """Start all detection modules."""
        self.running = True
        
        # Start process monitoring in a separate thread
        process_thread = threading.Thread(target=self._run_process_monitor)
        process_thread.daemon = True
        process_thread.start()
        
        # Start audio detection if available
        if self.speaker_detector:
            audio_thread = threading.Thread(target=self._run_audio_detection)
            audio_thread.daemon = True
            audio_thread.start()
        
        # Start site locker if enabled
        if self.site_locker:
            self.site_locker.start()
        
        # Start video processing
        self._run_video_processing()
        
    def stop(self):
        """Stop all detection modules."""
        self.running = False
        if self.site_locker:
            self.site_locker.stop()
        
    def _run_process_monitor(self):
        """Run the process monitor in a background thread."""
        while self.running:
            suspicious = self.process_monitor.scan_processes()
            if suspicious:
                print(f"[PROCESS MONITOR] Found {len(suspicious)} suspicious processes")
            time.sleep(30)  # Check every 30 seconds
    
    def _run_audio_detection(self):
        """Run the audio detection in a background thread."""
        if self.speaker_detector:
            self.speaker_detector.start()
            while self.running:
                # Audio detection runs in its own thread, we just monitor here
                time.sleep(1)
    
    def _run_video_processing(self):
        """Main video processing loop with all detections."""
        # Initialize video capture
        if self.input_source == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.input_source)
            
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
            
        # Set up display window
        cv2.namedWindow("Cheating Detection System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cheating Detection System", self.display_width, self.display_height)
        
        # Variables for blink detection (from detect_blinks.py)
        blink_thresh = 0.45
        succ_frame = 2
        count_frame = 0
        total_blinks = 0
        blink_state = False
        
        # Variables for gaze detection (from gaze_detector.py)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Mirror frame if webcam
                if self.input_source == 'webcam':
                    frame = cv2.flip(frame, 1)
                
                # Resize for processing
                frame = cv2.resize(frame, (640, 480))
                display_frame = frame.copy()
                
                # Convert to RGB for some detectors
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # ----------------------------
                # Run all detections
                # ----------------------------
                
                # 1. Blink detection (using dlib)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:
                    shape = landmark_predict(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    
                    lefteye = shape[L_start: L_end]
                    righteye = shape[R_start:R_end]
                    
                    left_EAR = calculate_EAR(lefteye)
                    right_EAR = calculate_EAR(righteye)
                    avg = (left_EAR + right_EAR) / 2
                    
                    if avg < blink_thresh:
                        count_frame += 1
                    else:
                        if count_frame >= succ_frame and not blink_state:
                            total_blinks += 1
                            blink_state = True
                        count_frame = 0
                        blink_state = False
                    
                    self.detection_states['blink']['count'] = total_blinks
                    self.detection_states['blink']['state'] = 'Closed' if avg < blink_thresh else 'Open'
                
                # 2. Gaze detection (using mediapipe)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        direction, right_pupil, left_pupil = get_gaze_direction(
                            frame, face_landmarks, frame.shape[1], frame.shape[0])
                        
                        self.detection_states['gaze']['direction'] = direction
                        self.detection_states['gaze']['looking_at_screen'] = direction == "center"
                        
                        # Draw gaze direction on frame
                        cv2.circle(display_frame, right_pupil, 3, (0, 0, 255), -1)
                        cv2.circle(display_frame, left_pupil, 3, (0, 0, 255), -1)
                
                # 3. Lip movement detection
                movement_detected, mouth_opened, openness, _ = self.lip_detector.process_frame(frame)
                self.detection_states['lips']['movement'] = movement_detected
                self.detection_states['lips']['open'] = mouth_opened
                self.detection_states['lips']['openness'] = openness
                
                # 4. Object detection (YOLO)
                annotated_frame, counts = self.yolo_detector.detect(frame)
                self.detection_states['objects']['counts'] = counts
                
                # Combine all detections into a single display frame
                self._update_display_frame(display_frame)
                
                # Show the combined frame
                cv2.imshow("Cheating Detection System", display_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop()
    
    def _update_display_frame(self, frame):
        """Update the display frame with all detection information."""
        # Create a black sidebar for status information
        sidebar = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
        
        # Add detection information to the sidebar
        y_offset = 30
        line_height = 30
        
        # Blink information
        blink_text = f"Blinks: {self.detection_states['blink']['count']}"
        state_text = f"Eyes: {self.detection_states['blink']['state']}"
        cv2.putText(sidebar, blink_text, (10, y_offset), self.font, self.font_scale, self.text_color, self.font_thickness)
        cv2.putText(sidebar, state_text, (10, y_offset + line_height), self.font, self.font_scale, 
                    (0, 0, 255) if self.detection_states['blink']['state'] == 'Closed' else (0, 255, 0), self.font_thickness)
        y_offset += line_height * 2
        
        # Gaze information
        gaze_text = f"Gaze: {self.detection_states['gaze']['direction']}"
        screen_text = "Looking at screen" if self.detection_states['gaze']['looking_at_screen'] else "Looking away!"
        cv2.putText(sidebar, gaze_text, (10, y_offset), self.font, self.font_scale, self.text_color, self.font_thickness)
        cv2.putText(sidebar, screen_text, (10, y_offset + line_height), self.font, self.font_scale, 
                    (0, 255, 0) if self.detection_states['gaze']['looking_at_screen'] else (0, 0, 255), self.font_thickness)
        y_offset += line_height * 2
        
        # Lip movement information
        lip_state = "OPEN" if self.detection_states['lips']['open'] else "MOVING" if self.detection_states['lips']['movement'] else "still"
        lip_color = (0, 255, 255) if self.detection_states['lips']['open'] else (0, 255, 0) if self.detection_states['lips']['movement'] else (0, 0, 255)
        lip_text = f"Lips: {lip_state}"
        openness_text = f"Openness: {self.detection_states['lips']['openness']:.2f}"
        cv2.putText(sidebar, lip_text, (10, y_offset), self.font, self.font_scale, lip_color, self.font_thickness)
        cv2.putText(sidebar, openness_text, (10, y_offset + line_height), self.font, self.font_scale * 0.8, self.text_color, self.font_thickness)
        y_offset += line_height * 2
        
        # Object detection information
        cv2.putText(sidebar, "Detected Objects:", (10, y_offset), self.font, self.font_scale, self.text_color, self.font_thickness)
        y_offset += line_height
        
        for obj, count in self.detection_states['objects']['counts'].items():
            if count > 0:
                obj_text = f"- {obj}: {count}"
                cv2.putText(sidebar, obj_text, (20, y_offset), self.font, self.font_scale * 0.8, 
                           (0, 0, 255) if obj in ['cell phone', 'laptop'] else self.text_color, self.font_thickness)
                y_offset += line_height
        
        # Audio detection information if available
        if self.speaker_detector:
            audio_text = f"Speakers: {self.detection_states['audio']['speakers']}"
            cv2.putText(sidebar, audio_text, (10, y_offset), self.font, self.font_scale, 
                       (0, 0, 255) if self.detection_states['audio']['speakers'] > 1 else (0, 255, 0), self.font_thickness)
            y_offset += line_height
        
        # Combine sidebar with main frame
        combined_frame = np.hstack((frame, sidebar))
        
        # Add warnings at the bottom of the frame
        warning_y = frame.shape[0] - 30
        if not self.detection_states['gaze']['looking_at_screen']:
            cv2.putText(combined_frame, "WARNING: Not looking at screen!", (10, warning_y), 
                        self.font, 0.7, self.alert_color, 2)
            warning_y -= 30
        
        if self.detection_states['lips']['open']:
            cv2.putText(combined_frame, "WARNING: Mouth open detected!", (10, warning_y), 
                        self.font, 0.7, self.alert_color, 2)
            warning_y -= 30
        
        if 'cell phone' in self.detection_states['objects']['counts'] and self.detection_states['objects']['counts']['cell phone'] > 0:
            cv2.putText(combined_frame, "WARNING: Cell phone detected!", (10, warning_y), 
                        self.font, 0.7, self.alert_color, 2)
            warning_y -= 30
        
        if self.detection_states['audio']['alert']:
            cv2.putText(combined_frame, "WARNING: Multiple speakers detected!", (10, warning_y), 
                        self.font, 0.7, self.alert_color, 2)
            warning_y -= 30
        
        return combined_frame

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Cheating Detection System")
    parser.add_argument('--input', type=str, default='webcam', help="Input source: 'webcam' or path to video/image file")
    parser.add_argument('--site-lock', action='store_true', help="Enable website locking")
    parser.add_argument('--allowed-url', type=str, help="URL to lock to if site locking is enabled")
    
    args = parser.parse_args()
    
    # Initialize and start the system
    system = CheatingDetectionSystem(
        input_source=args.input,
        enable_site_lock=args.site_lock,
        allowed_url=args.allowed_url
    )
    
    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()
        print("System stopped by user")

if __name__ == "__main__":
    main()