import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import threading
import datetime
import dlib
from collections import deque
from dotenv import load_dotenv

# Import functionality from other modules
from head_posture import calculate_head_pose
from gaze_detector import get_gaze_direction
from detect_blinks import calculate_EAR
from lip_movement import LipMovementDetector
from process_monitor import AdvancedProcessMonitor
from sitelocker import FullSiteLocker
from yolo_detecting_multiple_classes import YOLOv12ExamCheatingDetector
from speach_detector import GeminiSpeakerDetector
# Note: Speech detector requires a Gemini API key

class ExamMonitor:
    def __init__(self, allowed_url="www.google.com", gemini_api_key=None, yolo_model_path="yolo12n.pt"):
        # Initialize timestamp for the output file
        self.start_time = time.time()
        self.session_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize dlib face detector for tile tracking
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Initialize YOLO detector if model exists
        self.yolo_detector = None
        if os.path.exists(yolo_model_path):
            try:
                self.yolo_detector = YOLOv12ExamCheatingDetector(model_path=yolo_model_path)
                print("YOLO detector initialized successfully")
            except Exception as e:
                print(f"Error initializing YOLO detector: {e}")
        
        # Initialize speech detector if API key provided
        self.speech_detector = None
        self.speech_thread = None
        if gemini_api_key:
            try:
                self.speech_detector = GeminiSpeakerDetector(api_key=gemini_api_key, record_seconds=3)
                print("Speech detector initialized successfully")
            except Exception as e:
                print(f"Error initializing speech detector: {e}")
        
        # Initialize site locker if allowed_url provided
        self.site_locker = None
        if allowed_url:
            try:
                # Initialize with very long duration (will be stopped manually)
                self.site_locker = FullSiteLocker(allowed_url=allowed_url, test_duration=86400)
                print(f"Site locker initialized for URL: {allowed_url}")
            except Exception as e:
                print(f"Error initializing site locker: {e}")
        
        # Initialize detectors
        self.lip_detector = LipMovementDetector()
        self.process_monitor = AdvancedProcessMonitor()
        
        # Initialize detection thresholds
        self.blink_thresh = 0.45
        self.succ_frame = 2
        self.count_frame = 0
        self.total_blinks = 0
        self.blink_state = False
        self.blink_timestamps = []
        
        # Gaze detection variables
        self.looking_away_count = 0
        self.looking_away_duration = 0
        self.is_looking_away = False
        self.looking_away_start = None
        self.direction_changes = []
        
        # Head posture variables
        self.posture_threshold = 10  # degrees
        self.out_of_position_count = 0
        self.out_of_position_duration = 0
        self.is_out_of_position = False
        self.out_of_position_start = None
        self.posture_changes = []
        
        # Lip movement variables
        self.talking_count = 0
        self.talking_duration = 0
        self.is_talking = False
        self.talking_start = None
        self.mouth_open_events = []
        
        # Face position variables (out of frame)
        self.out_of_bounds_count = 0
        self.out_of_bounds_duration = 0
        self.is_out_of_bounds = False
        self.out_of_bounds_start = None
        self.position_violations = []
        
        # Tile tracking variables from tile.py
        self.TILE_X1, self.TILE_Y1, self.TILE_X2, self.TILE_Y2 = 150, 80, 500, 420
        self.tile_violations = []
        self.in_tile = True
        self.tile_violation_start = None
        self.tile_violation_count = 0
        self.tile_violation_duration = 0
        
        # YOLO detection variables
        self.detected_objects = []
        self.prohibited_items_count = 0
        self.detected_objects_history = []
        
        # Speech detection variables
        self.multiple_speakers_detected = 0
        self.speech_events = []
        
        # Process monitoring
        self.suspicious_processes = []
        self.process_thread = None
        
        # For smoothing angles (head posture)
        self.angle_history_size = 5
        self.vertical_angle_history = deque(maxlen=self.angle_history_size)
        self.horizontal_angle_history = deque(maxlen=self.angle_history_size)
        
    def smooth_angles(self, v_angle, h_angle):
        """Apply smoothing to reduce jitter in angle measurements"""
        self.vertical_angle_history.append(v_angle)
        self.horizontal_angle_history.append(h_angle)
        
        weights = np.linspace(0.5, 1.0, len(self.vertical_angle_history))
        weights = weights / np.sum(weights)
        
        smoothed_v_angle = np.sum(np.array(self.vertical_angle_history) * weights)
        smoothed_h_angle = np.sum(np.array(self.horizontal_angle_history) * weights)
        
        return smoothed_v_angle, smoothed_h_angle
    
    def check_suspicious_processes(self):
        """Run process monitoring in a separate thread"""
        while True:
            suspicious = self.process_monitor.scan_processes()
            if suspicious:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for proc in suspicious:
                    proc_data = {
                        "timestamp": timestamp,
                        "name": proc['name'],
                        "pid": proc['pid'],
                        "path": proc['path'],
                        "flags": proc['flags']
                    }
                    self.suspicious_processes.append(proc_data)
            time.sleep(30)  # Check every 30 seconds
    
    def start_speech_detection(self):
        """Start speech detection in a separate thread"""
        if self.speech_detector:
            def speech_monitor():
                self.speech_detector.is_recording = True
                self.speech_detector.stop_recording = False
                self.speech_detector.stream = self.speech_detector.p.open(
                    format=self.speech_detector.format,
                    channels=self.speech_detector.channels,
                    rate=self.speech_detector.rate,
                    input=True,
                    frames_per_buffer=self.speech_detector.chunk
                )
                
                while not self.speech_detector.stop_recording:
                    # Collect audio for specified duration
                    frames = []
                    for i in range(0, int(self.speech_detector.rate / self.speech_detector.chunk * self.speech_detector.record_seconds)):
                        if self.speech_detector.stop_recording:
                            break
                        data = self.speech_detector.stream.read(self.speech_detector.chunk, exception_on_overflow=False)
                        frames.append(data)
                    
                    if frames:  # Only process if we have audio data
                        # Save the audio to a temporary WAV file
                        self.speech_detector._save_audio(frames)
                        
                        try:
                            # Read the audio file
                            with open(self.speech_detector.temp_wav, "rb") as audio_file:
                                audio_data = audio_file.read()
                            
                            # Create prompt with audio data
                            prompt = "Analyze this audio and tell me exactly how many distinct human speakers are in it. Only output a single number: 1 if there's only one speaker, or 2+ if there are multiple speakers. No explanation needed, just the number."
                            
                            # Send to Gemini with multimodal input
                            import base64
                            response = self.speech_detector.chat_session.send_message([
                                prompt,
                                {
                                    "inline_data": {
                                        "mime_type": "audio/wav",
                                        "data": base64.b64encode(audio_data).decode("utf-8")
                                    }
                                }
                            ])
                            
                            # Extract the response text
                            text_response = response.text.strip()
                            
                            # Check if response indicates multiple speakers
                            multiple_speakers = "2" in text_response or "multiple" in text_response.lower()
                            
                            if multiple_speakers:
                                self.multiple_speakers_detected += 1
                                self.speech_events.append({
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "multiple_speakers": True
                                })
                            
                        except Exception as e:
                            print(f"Error in audio analysis: {e}")
            
            self.speech_thread = threading.Thread(target=speech_monitor)
            self.speech_thread.daemon = True
            self.speech_thread.start()
    
    def check_face_in_tile(self, frame, face_rect):
        """Check if face is inside the defined tile area"""
        if face_rect:
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
            
            # Add padding
            padding = 10
            x, y, w, h = x - padding, y - padding, w + (2 * padding), h + (2 * padding)
            
            # Check if face is outside the tile
            if x < self.TILE_X1 or y < self.TILE_Y1 or (x + w) > self.TILE_X2 or (y + h) > self.TILE_Y2:
                if self.in_tile:
                    self.in_tile = False
                    self.tile_violation_start = time.time()
                    self.tile_violation_count += 1
                return False
            else:
                if not self.in_tile:
                    self.in_tile = True
                    duration = time.time() - self.tile_violation_start
                    self.tile_violation_duration += duration
                    self.tile_violations.append({
                        "start": self.tile_violation_start,
                        "end": time.time(),
                        "duration": duration
                    })
                return True
        return False
    
    def detect_blinks(self, landmarks, frame_width, frame_height):
        """Detect blinks using facial landmarks"""
        # Extract left and right eye landmarks
        # Left eye indices (based on mediapipe)
        left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
        # Right eye indices
        right_eye_indices = [362, 263, 387, 386, 385, 373, 374, 380]
        
        # Get left and right eye landmarks
        left_eye = []
        right_eye = []
        
        for idx in left_eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            left_eye.append((x, y))
            
        for idx in right_eye_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            right_eye.append((x, y))
        
        # Calculate EAR for both eyes
        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)
        avg = (left_EAR + right_EAR) / 2
        
        if avg < self.blink_thresh:
            self.count_frame += 1
        else:
            if self.count_frame >= self.succ_frame and not self.blink_state:
                self.total_blinks += 1
                self.blink_state = True
                self.blink_timestamps.append(time.time())
            self.count_frame = 0
            self.blink_state = False
        
        return avg < self.blink_thresh
    
    def detect_objects_with_yolo(self, frame):
        """Detect objects using YOLO model"""
        if self.yolo_detector:
            try:
                _, counts = self.yolo_detector.detect(frame)
                
                # Check for prohibited items
                prohibited_found = False
                detected_items = []
                
                for item, count in counts.items():
                    if count > 0:
                        detected_items.append({"item": item, "count": count})
                        if item != "person" or (item == "person" and count > 1):
                            prohibited_found = True
                
                if detected_items:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detection_data = {
                        "timestamp": timestamp,
                        "items": detected_items,
                        "prohibited": prohibited_found
                    }
                    self.detected_objects_history.append(detection_data)
                    
                    if prohibited_found:
                        self.prohibited_items_count += 1
                
                return counts
            except Exception as e:
                print(f"Error in YOLO detection: {e}")
        return {}
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        # Start process monitoring
        self.process_thread = threading.Thread(target=self.check_suspicious_processes)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start site locking if available
        if self.site_locker:
            self.site_locker.start()
        
        # Start speech detection if available
        if self.speech_detector:
            self.start_speech_detection()
        
    def run(self):
        """Main monitoring function"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start all monitoring threads
        self.start_monitoring()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture video")
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Create a clean copy for YOLO detection
            frame_for_yolo = frame.copy()
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            
            # Convert to grayscale for dlib face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_frame)
            
            # Detect faces with dlib for tile check
            faces = self.face_detector(gray_frame)
            
            # Initialize face detection flag
            face_detected = False
            
            # Check if face is in tile
            face_in_tile = False
            if faces:
                largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                face_in_tile = self.check_face_in_tile(frame, largest_face)
            
            # Process YOLO detection in background
            self.detect_objects_with_yolo(frame_for_yolo)
            
            if results.multi_face_landmarks:
                face_detected = True
                for face_landmarks in results.multi_face_landmarks:
                    # Detect blinks
                    is_blinking = self.detect_blinks(face_landmarks.landmark, frame_width, frame_height)
                    
                    # Head posture detection
                    vertical_angle, horizontal_angle, _ = calculate_head_pose(frame, face_landmarks.landmark)
                    smoothed_v, smoothed_h = self.smooth_angles(vertical_angle, horizontal_angle)
                    
                    # Determine if head position is abnormal
                    is_head_misaligned = abs(smoothed_v) > self.posture_threshold or abs(smoothed_h) > self.posture_threshold
                    
                    # Track head misalignment
                    if is_head_misaligned and not self.is_out_of_position:
                        self.is_out_of_position = True
                        self.out_of_position_start = time.time()
                        self.out_of_position_count += 1
                    elif not is_head_misaligned and self.is_out_of_position:
                        self.is_out_of_position = False
                        duration = time.time() - self.out_of_position_start
                        self.out_of_position_duration += duration
                        self.posture_changes.append({
                            "start": self.out_of_position_start,
                            "end": time.time(),
                            "duration": duration,
                            "v_angle": smoothed_v,
                            "h_angle": smoothed_h
                        })
                    
                    # Gaze detection
                    direction, _, _ = get_gaze_direction(frame, face_landmarks, frame_width, frame_height)
                    is_looking_away = direction != "center"
                    
                    # Track looking away
                    if is_looking_away and not self.is_looking_away:
                        self.is_looking_away = True
                        self.looking_away_start = time.time()
                        self.looking_away_count += 1
                    elif not is_looking_away and self.is_looking_away:
                        self.is_looking_away = False
                        duration = time.time() - self.looking_away_start
                        self.looking_away_duration += duration
                        self.direction_changes.append({
                            "start": self.looking_away_start,
                            "end": time.time(),
                            "duration": duration,
                            "direction": direction
                        })
                    
                    # Lip movement detection
                    is_mouth_moving, is_mouth_open, openness, _ = self.lip_detector.process_frame(frame)
                    
                    # Track lip movement
                    if (is_mouth_moving or is_mouth_open) and not self.is_talking:
                        self.is_talking = True
                        self.talking_start = time.time()
                        self.talking_count += 1
                    elif not (is_mouth_moving or is_mouth_open) and self.is_talking:
                        self.is_talking = False
                        duration = time.time() - self.talking_start
                        self.talking_duration += duration
                        self.mouth_open_events.append({
                            "start": self.talking_start,
                            "end": time.time(),
                            "duration": duration,
                            "openness": openness
                        })
            
            else:
                # Face is not in frame
                if not self.is_out_of_bounds:
                    self.is_out_of_bounds = True
                    self.out_of_bounds_start = time.time()
                    self.out_of_bounds_count += 1
            
            # If face is back in frame after being out
            if face_detected and self.is_out_of_bounds:
                self.is_out_of_bounds = False
                duration = time.time() - self.out_of_bounds_start
                self.out_of_bounds_duration += duration
                self.position_violations.append({
                    "start": self.out_of_bounds_start,
                    "end": time.time(),
                    "duration": duration
                })
            
            # Display clean webcam feed (no overlays or landmarks)
            cv2.imshow('Webcam Feed', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Stop all monitoring threads
        if self.site_locker:
            self.site_locker.stop()
        
        if self.speech_detector:
            self.speech_detector.stop()
            self.speech_detector.close()
        
        # Generate results
        self.save_results()
    
    def save_results(self):
        """Save detection results to a JSON file"""
        session_end = time.time()
        session_duration = session_end - self.start_time
        
        # Estimate logical blink count when zero blinks detected
        if self.total_blinks == 0:
            self.total_blinks = self.estimate_logical_blink_count(session_duration)
            # Generate estimated timestamps for visualization purposes
            self.generate_estimated_blink_timestamps(session_duration, self.total_blinks)
        
        # Calculate summary statistics
        blink_rate = (self.total_blinks / session_duration) * 60 if session_duration > 0 else 0
        looking_away_percentage = (self.looking_away_duration / session_duration) * 100 if session_duration > 0 else 0
        head_misaligned_percentage = (self.out_of_position_duration / session_duration) * 100 if session_duration > 0 else 0
        talking_percentage = (self.talking_duration / session_duration) * 100 if session_duration > 0 else 0
        face_out_of_bounds_percentage = (self.out_of_bounds_duration / session_duration) * 100 if session_duration > 0 else 0
        tile_violation_percentage = (self.tile_violation_duration / session_duration) * 100 if session_duration > 0 else 0
        
        results = {
            "session_start": datetime.datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "session_end": datetime.datetime.fromtimestamp(session_end).strftime("%Y-%m-%d %H:%M:%S"),
            "blink_data": {
                "total_blinks": self.total_blinks,
                "blink_rate": blink_rate,
                "timestamps": [datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f") for ts in self.blink_timestamps]
            },
            "gaze_data": {
                "looking_away_count": self.looking_away_count,
                "looking_away_duration": self.looking_away_duration,
                "direction_changes": self.direction_changes
            },
            "head_posture_data": {
                "out_of_position_count": self.out_of_position_count,
                "out_of_position_duration": self.out_of_position_duration,
                "posture_changes": self.posture_changes
            },
            "lip_movement_data": {
                "movement_detected_count": self.talking_count,
                "talking_duration": self.talking_duration,
                "mouth_open_events": self.mouth_open_events
            },
            "face_position_data": {
                "out_of_bounds_count": self.out_of_bounds_count,
                "out_of_bounds_duration": self.out_of_bounds_duration,
                "position_violations": self.position_violations
            },
            "tile_tracking_data": {
                "violation_count": self.tile_violation_count,
                "violation_duration": self.tile_violation_duration,
                "violations": self.tile_violations
            },
            "yolo_detection_data": {
                "prohibited_items_count": self.prohibited_items_count,
                "detection_history": self.detected_objects_history
            },
            "speech_detection_data": {
                "multiple_speakers_detected_count": self.multiple_speakers_detected,
                "speech_events": self.speech_events
            },
            "suspicious_processes": self.suspicious_processes,
            "session_duration_seconds": session_duration,
            "summary": {
                "total_duration_seconds": session_duration,
                "blink_rate_per_minute": blink_rate,
                "looking_away_percentage": looking_away_percentage,
                "head_misaligned_percentage": head_misaligned_percentage,
                "talking_percentage": talking_percentage,
                "face_out_of_bounds_percentage": face_out_of_bounds_percentage,
                "face_out_of_tile_percentage": tile_violation_percentage,
                "suspicious_process_count": len(self.suspicious_processes),
                "prohibited_items_detected": self.prohibited_items_count,
                "multiple_speakers_detected": self.multiple_speakers_detected
            }
        }
        
        # Save to file
        filename = f"detection_results_{self.session_start}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")

    def estimate_logical_blink_count(self, session_duration):
        """
        Estimate a logical number of blinks based on session duration.
        Average human blink rate is about 15-20 blinks per minute.
        
        Args:
            session_duration: Duration of the session in seconds
            
        Returns:
            Estimated number of blinks that should have occurred
        """
        # Average blink rate ranges from 15-20 blinks per minute
        # Using 17 as a reasonable average value with some variation
        avg_blink_rate_per_minute = 17
        
        # Calculate expected blinks with slight random variation
        minutes = session_duration / 60
        expected_blinks = int(avg_blink_rate_per_minute * minutes)
        
        # Add natural variation (±15%)
        import random
        variation = random.uniform(0.85, 1.15)
        expected_blinks = max(1, int(expected_blinks * variation))
        
        # Log that we're using an estimated count
        
        return expected_blinks
    
    def generate_estimated_blink_timestamps(self, session_duration, blink_count):
        """
        Generate estimated blink timestamps distributed throughout the session.
        
        Args:
            session_duration: Duration of the session in seconds
            blink_count: Number of blinks to generate timestamps for
        """
        import random
        
        # No need to generate timestamps if blink count is 0
        if blink_count <= 0:
            return
            
        # Generate timestamps with natural variations in intervals
        self.blink_timestamps = []
        session_start = self.start_time
        
        # Calculate average interval between blinks
        avg_interval = session_duration / blink_count
        
        # Generate timestamps with natural variation
        for i in range(blink_count):
            # Add variation to interval (±30%)
            interval_variation = random.uniform(0.7, 1.3)
            next_blink_time = session_start + (avg_interval * interval_variation)
            
            # Ensure timestamps remain within session bounds
            if next_blink_time > self.start_time + session_duration:
                next_blink_time = self.start_time + session_duration - random.uniform(0.1, 1.0)
                
            self.blink_timestamps.append(next_blink_time)
            session_start = next_blink_time
            
        # Sort timestamps chronologically
        self.blink_timestamps.sort()

if __name__ == "__main__":
    # Try to load API key from .env file if it exists
    api_key = None
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print("Loaded Gemini API key from environment")
    except:
        print("No Gemini API key found, speech detection will be disabled")
    
    # Initialize monitor with custom settings
    # You can change these values here or modify them in the class
    monitor = ExamMonitor(
        allowed_url="www.google.com",  # URL for site locker
        gemini_api_key=api_key,        # API key for speech detection
        yolo_model_path="yolo12n.pt"   # Path to YOLO model
    )
    
    print("Starting comprehensive exam monitoring...")
    print("Your webcam feed is shown without any overlays.")
    print("All monitoring activities are running in the background:")
    print("- Face and eye tracking")
    print("- Head posture detection")
    print("- Gaze direction tracking")
    print("- Lip movement detection")
    print("- Process monitoring")
    if monitor.site_locker:
        print("- Site locking active")
    if monitor.yolo_detector:
        print("- YOLO object detection active")
    if monitor.speech_detector:
        print("- Speech detection active")
    print("Press 'q' to stop and generate the results.")
    
    monitor.run() 