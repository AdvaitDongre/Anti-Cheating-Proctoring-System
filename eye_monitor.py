import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from collections import deque
from scipy import signal

class EyeMonitor:
    def __init__(self, shape_predictor_path, blink_threshold=0.005, blink_consec_frames=1, 
                 gaze_threshold=0.35, direction_time_threshold=3.0, 
                 blink_rate_low=10, blink_rate_high=30):
        """
        Initialize the EyeMonitor class with improved blink detection.
        
        Parameters:
        - shape_predictor_path: Path to dlib's facial landmark predictor
        - blink_threshold: Eye aspect ratio threshold for blink detection
        - blink_consec_frames: Number of consecutive frames for blink confirmation
        - gaze_threshold: Threshold for determining gaze direction
        - direction_time_threshold: Time threshold (in seconds) for gaze direction warning
        - blink_rate_low: Lower threshold for normal blink rate (blinks per minute)
        - blink_rate_high: Upper threshold for normal blink rate (blinks per minute)
        """
        # Initialize face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        
        # Define eye landmark indices
        self.lStart, self.lEnd = 42, 48  # Left eye landmarks
        self.rStart, self.rEnd = 36, 42  # Right eye landmarks
        
        # Set thresholds
        self.EYE_AR_THRESH = blink_threshold
        self.EYE_AR_CONSEC_FRAMES = blink_consec_frames
        self.GAZE_THRESHOLD = gaze_threshold
        self.DIRECTION_TIME_THRESHOLD = direction_time_threshold
        self.BLINK_RATE_LOW = blink_rate_low
        self.BLINK_RATE_HIGH = blink_rate_high
        
        # Initialize counters and tracking variables
        self.counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        self.start_time = time.time()
        self.blink_rate = 0
        
        # Improved tracking for blink state machine
        self.BLINK_STATE_OPEN = 0
        self.BLINK_STATE_CLOSING = 1
        self.BLINK_STATE_CLOSED = 2
        self.BLINK_STATE_OPENING = 3
        self.blink_state = self.BLINK_STATE_OPEN
        
        # Minimum time between blinks (in seconds)
        self.MIN_BLINK_INTERVAL = 0.15
        
        # Store baseline EAR for calibration
        self.baseline_ear = None
        self.baseline_samples = []
        self.calibrated = False
        self.calibration_frames = 60  # Increased for better calibration
        
        # For gaze direction tracking
        self.gaze_direction = "center"
        self.direction_start_time = time.time()
        self.direction_warning = False
        
        # For storing blink history (last 60 seconds)
        self.blink_times = deque(maxlen=200)  # Increased capacity
        self.blink_rate_warning = False
        
        # Enhanced EAR tracking with separate histories for different purposes
        self.ear_history_short = deque(maxlen=3)  # For quick response to blinks
        self.ear_history_medium = deque(maxlen=10)  # For smoothing display values
        self.ear_history_long = deque(maxlen=30)  # For establishing trends
        
        # Track min and max EAR for adaptive thresholds
        self.ear_min = 1.0
        self.ear_max = 0.0
        
        # Track complete blink profiles
        self.blink_profile_samples = []
        self.blink_duration_history = deque(maxlen=10)
        self.avg_blink_duration = 0.3  # Default starting value (seconds)
        
        # Blink velocity tracking (rate of change in EAR)
        self.ear_velocity_history = deque(maxlen=5)
        self.last_ear = None
        self.last_velocity_time = time.time()
        
        # Signal processing parameters
        self.signal_filter_size = 3  # Size of median filter
        self.filter_buffer = deque(maxlen=self.signal_filter_size)
        
        # Adaptive thresholds
        self.adaptive_thresh_enabled = False
        self.ear_threshold_ratio = 0.75  # Initial ratio of baseline for threshold
        
        # Debug variables
        self.debug_ear_min = 1.0
        self.debug_ear_max = 0.0
        self.debug_velocity = 0.0
        self.debug_blink_duration = 0.0
        self.blink_start_time = None
        
        print("Enhanced EyeMonitor initialized with the following parameters:")
        print(f"- Initial blink threshold: {self.EYE_AR_THRESH}")
        print(f"- Consecutive frames: {self.EYE_AR_CONSEC_FRAMES}")
        print(f"- Adaptive thresholding: {'Enabled' if self.adaptive_thresh_enabled else 'Disabled'}")
    
    def eye_aspect_ratio(self, eye):
        """
        Calculate the eye aspect ratio with additional sanity checks.
        """
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        
        # Avoid division by zero and unrealistic values
        if C < 0.1:
            return 0
            
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        # Sanity check - EAR should typically be between 0.1 and 0.5
        if ear < 0.05 or ear > 0.7:
            # If we have history, return the last valid value
            if self.ear_history_medium:
                return np.mean(self.ear_history_medium)
            return 0.25  # Default fallback value
            
        return ear
    
    def filter_ear(self, ear):
        """
        Apply signal processing to filter the EAR value.
        """
        # Add to filter buffer
        self.filter_buffer.append(ear)
        
        # Apply median filter to remove outliers
        if len(self.filter_buffer) == self.signal_filter_size:
            filtered_ear = np.median(self.filter_buffer)
        else:
            filtered_ear = ear
        
        return filtered_ear
    
    def update_ear_velocity(self, ear, timestamp):
        """
        Calculate and track the velocity (rate of change) of the EAR.
        """
        if self.last_ear is not None:
            time_delta = timestamp - self.last_velocity_time
            if time_delta > 0:
                velocity = (ear - self.last_ear) / time_delta
                self.ear_velocity_history.append(velocity)
                self.debug_velocity = velocity
        
        self.last_ear = ear
        self.last_velocity_time = timestamp
    
    def get_smoothed_ear(self, ear, timestamp):
        """
        Apply smoothing to the EAR value with enhanced tracking.
        """
        # First apply signal filtering
        filtered_ear = self.filter_ear(ear)
        
        # Update velocity tracking
        self.update_ear_velocity(filtered_ear, timestamp)
        
        # Update different time-scale histories
        self.ear_history_short.append(filtered_ear)
        self.ear_history_medium.append(filtered_ear)
        self.ear_history_long.append(filtered_ear)
        
        # Track min/max for adaptive thresholds
        if filtered_ear > 0.1:  # Ignore potential errors
            if filtered_ear > self.ear_max:
                self.ear_max = filtered_ear
            if filtered_ear < self.ear_min:
                self.ear_min = filtered_ear
            
            # Also update debug values
            if filtered_ear > self.debug_ear_max:
                self.debug_ear_max = filtered_ear
            if filtered_ear < self.debug_ear_min:
                self.debug_ear_min = filtered_ear
        
        # Return short-term average for blink detection (more responsive)
        return np.mean(self.ear_history_short) if self.ear_history_short else filtered_ear
    
    def update_adaptive_threshold(self):
        """
        Update the blink detection threshold based on observed EAR statistics.
        """
        if not self.adaptive_thresh_enabled or not self.calibrated:
            return
        
        # Calculate threshold as a fraction of the baseline
        # But also incorporate knowledge of the observed min/max range
        if self.baseline_ear and self.ear_min < self.baseline_ear:
            # Calculate the blink threshold as a point between min and baseline
            # This is more robust to different eye shapes and lighting conditions
            threshold_range = self.baseline_ear - self.ear_min
            # Use 30% above the minimum as threshold
            new_threshold = self.ear_min + (threshold_range * 0.3)
            
            # Apply smoothing to avoid threshold jumps
            self.EYE_AR_THRESH = self.EYE_AR_THRESH * 0.7 + new_threshold * 0.3
            
            # Make sure threshold isn't too high or too low
            if self.EYE_AR_THRESH > self.baseline_ear * 0.9:
                self.EYE_AR_THRESH = self.baseline_ear * 0.9
            elif self.EYE_AR_THRESH < self.baseline_ear * 0.65:
                self.EYE_AR_THRESH = self.baseline_ear * 0.65
    
    def calibrate_ear(self, ear):
        """
        Enhanced calibration of the baseline EAR value for the current user
        """
        if not self.calibrated:
            # Only add valid EAR values to calibration samples
            if 0.1 < ear < 0.5:  # Filter out abnormal values
                self.baseline_samples.append(ear)
            
            if len(self.baseline_samples) >= self.calibration_frames:
                # Remove outliers using quantile-based filtering
                sorted_samples = sorted(self.baseline_samples)
                q25 = int(len(sorted_samples) * 0.25)
                q75 = int(len(sorted_samples) * 0.75)
                
                # Use the interquartile range for a more robust baseline
                valid_samples = sorted_samples[q25:q75]
                
                self.baseline_ear = np.mean(valid_samples)
                self.calibrated = True
                
                # Set initial threshold based on baseline
                self.EYE_AR_THRESH = self.baseline_ear * self.ear_threshold_ratio
                
                # Also initialize min/max values
                self.ear_min = min(sorted_samples)
                self.ear_max = max(sorted_samples)
                
                print(f"Enhanced calibration complete.")
                print(f"Baseline EAR: {self.baseline_ear:.3f}")
                print(f"Initial Threshold: {self.EYE_AR_THRESH:.3f}")
                print(f"Min EAR observed: {self.ear_min:.3f}")
                print(f"Max EAR observed: {self.ear_max:.3f}")
        
        return self.calibrated
    
    def detect_blink(self, ear, timestamp):
        """
        Enhanced blink detection using a state machine approach.
        Returns True if a complete blink is detected.
        """
        # If not calibrated, we can't reliably detect blinks
        if not self.calibrated:
            return False
        
        # Check if we have sufficient time since last blink to avoid duplicates
        if self.total_blinks > 0 and timestamp - self.last_blink_time < self.MIN_BLINK_INTERVAL:
            return False
        
        # Get velocity information (rate of EAR change)
        velocity = np.mean(self.ear_velocity_history) if self.ear_velocity_history else 0
        
        # State machine for accurate blink detection
        blink_completed = False
        
        # State transition logic
        if self.blink_state == self.BLINK_STATE_OPEN:
            # Eye is currently open, check if starting to close
            if ear < self.EYE_AR_THRESH:
                self.blink_state = self.BLINK_STATE_CLOSING
                self.counter = 1
                self.blink_start_time = timestamp
            
        elif self.blink_state == self.BLINK_STATE_CLOSING:
            # Eye is in the process of closing
            if ear < self.EYE_AR_THRESH:
                self.counter += 1
                # If eye has been closing for enough frames, consider it closed
                if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.blink_state = self.BLINK_STATE_CLOSED
            else:
                # If EAR goes back above threshold, eye didn't actually close
                # So go back to open state
                self.blink_state = self.BLINK_STATE_OPEN
                self.counter = 0
                
        elif self.blink_state == self.BLINK_STATE_CLOSED:
            # Eye is currently closed, check if starting to open
            if ear > self.EYE_AR_THRESH:
                self.blink_state = self.BLINK_STATE_OPENING
            
        elif self.blink_state == self.BLINK_STATE_OPENING:
            # Eye is in the process of opening
            # More stringent condition for complete opening to avoid false reopens
            if ear > (self.EYE_AR_THRESH * 1.1):
                self.blink_state = self.BLINK_STATE_OPEN
                blink_completed = True
                
                # Calculate blink duration
                if self.blink_start_time is not None:
                    self.debug_blink_duration = timestamp - self.blink_start_time
                    self.blink_duration_history.append(self.debug_blink_duration)
                    self.avg_blink_duration = np.mean(self.blink_duration_history)
            
        # Return whether a complete blink was detected
        return blink_completed
        
    def determine_gaze_direction(self, left_eye, right_eye):
        """Determine the gaze direction based on eye position"""
        # Calculate the center of each eye
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)
        
        # Calculate the position of the iris (approximation using the eye landmarks)
        # We use points 1, 2 vs 4, 5 to estimate where the iris is positioned horizontally
        left_inner = (left_eye[1] + left_eye[2]) / 2
        left_outer = (left_eye[4] + left_eye[5]) / 2
        right_inner = (right_eye[1] + right_eye[2]) / 2
        right_outer = (right_eye[4] + right_eye[5]) / 2
        
        # Use relative positions to determine gaze direction
        left_ratio = dist.euclidean(left_inner, left_eye[0]) / dist.euclidean(left_outer, left_eye[3])
        right_ratio = dist.euclidean(right_inner, right_eye[0]) / dist.euclidean(right_outer, right_eye[3])
        
        # Average the ratios
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Determine gaze direction
        if avg_ratio < 1 - self.GAZE_THRESHOLD:
            return "left"
        elif avg_ratio > 1 + self.GAZE_THRESHOLD:
            return "right"
        else:
            return "center"
    
    def calculate_blink_rate(self):
        """Calculate the current blink rate (blinks per minute)"""
        current_time = time.time()
        
        # Remove blinks older than 60 seconds
        while self.blink_times and current_time - self.blink_times[0] > 60:
            self.blink_times.popleft()
        
        # Calculate blink rate based on blinks in the last minute
        elapsed = min(60, current_time - self.start_time)
        if elapsed > 0:
            self.blink_rate = len(self.blink_times) * (60 / elapsed)
        else:
            self.blink_rate = 0
        
        return self.blink_rate
    
    def check_blink_rate_warning(self):
        """Check if the blink rate is abnormal"""
        # Only check after we have enough data (at least 30 seconds)
        if time.time() - self.start_time > 30:
            if self.blink_rate < self.BLINK_RATE_LOW or self.blink_rate > self.BLINK_RATE_HIGH:
                self.blink_rate_warning = True
            else:
                self.blink_rate_warning = False
        
        return self.blink_rate_warning
    
    def process_frame(self, frame, face_box=None):
        """
        Process a video frame to detect eyes, blinks, and gaze.
        
        Args:
            frame: Input video frame
            face_box: Optional tuple (x, y, w, h) of face coordinates from external detector
            
        Returns:
        - processed_frame: Frame with annotations
        - metrics: Dictionary containing eye metrics and warnings
        """
        # Create a copy to avoid modifying the original frame
        processed_frame = frame.copy()
        
        # Current timestamp for timing operations
        current_time = time.time()
        
        # Resize frame for faster processing (don't modify original)
        process_frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Initialize metrics dictionary
        metrics = {
            "ear": 0,
            "blinks": self.total_blinks,
            "blink_rate": self.blink_rate,
            "gaze_direction": "unknown",
            "direction_warning": False,
            "blink_rate_warning": False,
            "face_detected": False,
            "calibrated": self.calibrated,
            "blink_state": self.blink_state  # New metric for debugging
        }
        
        # If face_box is provided, convert it to dlib rectangle
        if face_box is not None:
            x, y, w, h = face_box
            # Scale coordinates to processing frame size
            scale_x = 640 / frame.shape[1]
            scale_y = 480 / frame.shape[0]
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            face = dlib.rectangle(x, y, x + w, y + h)
            faces = [face]
            metrics["face_detected"] = True
        else:
            # Detect faces using dlib if no face_box provided
            faces = self.detector(gray)
            metrics["face_detected"] = len(faces) > 0
        
        if len(faces) > 0:
            face = faces[0]  # Process the first face
            
            # Scale the face coordinates to original frame size if needed
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            
            # Get facial landmarks
            shape = self.predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Extract the left and right eye coordinates
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            
            # Calculate the eye aspect ratios
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            
            # Average the eye aspect ratio for both eyes
            raw_ear = (leftEAR + rightEAR) / 2.0
            
            # Apply smoothing and filtering
            ear = self.get_smoothed_ear(raw_ear, current_time)
            metrics["ear"] = ear
            
            # Scale eye coordinates to original frame size
            scaled_leftEye = [[int(x * scale_x), int(y * scale_y)] for x, y in leftEye]
            scaled_rightEye = [[int(x * scale_x), int(y * scale_y)] for x, y in rightEye]
            
            # Add face box to metrics
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            metrics["face_box"] = (int(x * scale_x), int(y * scale_y), 
                                  int(w * scale_x), int(h * scale_y))
            
            # Handle calibration
            if not self.calibrated:
                is_calibrated = self.calibrate_ear(ear)
                if is_calibrated:
                    metrics["calibrated"] = True
                    
                # Show calibration message on frame
                cv2.putText(processed_frame, "Calibrating... Please keep eyes open", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Progress: {len(self.baseline_samples)}/{self.calibration_frames}", 
                           (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Update adaptive threshold if enabled
                self.update_adaptive_threshold()
                
                # Detect blinks with enhanced algorithm
                blink_detected = self.detect_blink(ear, current_time)
                
                if blink_detected:
                    self.total_blinks += 1
                    
                    # Add blink timestamp to history
                    self.blink_times.append(current_time)
                    self.last_blink_time = current_time
                    
                    # Update metrics
                    metrics["blinks"] = self.total_blinks
                    print(f"Blink detected! Total: {self.total_blinks}, Duration: {self.debug_blink_duration:.3f}s")
            
            # Draw the contours of the eyes on the original sized frame
            leftEyeHull = cv2.convexHull(np.array(scaled_leftEye))
            rightEyeHull = cv2.convexHull(np.array(scaled_rightEye))
            
            # Color based on blink state
            if self.blink_state == self.BLINK_STATE_OPEN:
                eye_color = (0, 255, 0)  # Green for open
            elif self.blink_state == self.BLINK_STATE_CLOSING or self.blink_state == self.BLINK_STATE_OPENING:
                eye_color = (255, 165, 0)  # Orange for transition
            else:
                eye_color = (0, 0, 255)  # Red for closed
                
            cv2.drawContours(processed_frame, [leftEyeHull], -1, eye_color, 1)
            cv2.drawContours(processed_frame, [rightEyeHull], -1, eye_color, 1)
            
            # Determine gaze direction
            current_direction = self.determine_gaze_direction(leftEye, rightEye)
            metrics["gaze_direction"] = current_direction
            
            # Check if gaze direction has changed
            if current_direction != self.gaze_direction:
                self.gaze_direction = current_direction
                self.direction_start_time = current_time
                self.direction_warning = False
            
            # Check if looking in the same direction for too long
            if current_direction != "center" and current_time - self.direction_start_time > self.DIRECTION_TIME_THRESHOLD:
                self.direction_warning = True
            
            metrics["direction_warning"] = self.direction_warning
            
            # Calculate current blink rate
            self.calculate_blink_rate()
            metrics["blink_rate"] = self.blink_rate
            
            # Check for abnormal blink rate
            self.check_blink_rate_warning()
            metrics["blink_rate_warning"] = self.blink_rate_warning
            
            # Draw info and status on frame
            ear_color = (0, 0, 255) if ear < self.EYE_AR_THRESH else (0, 255, 0)
            cv2.putText(processed_frame, f"EAR: {ear:.3f}", (frame.shape[1] - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            
            # Draw threshold line
            if self.calibrated:
                cv2.putText(processed_frame, f"Threshold: {self.EYE_AR_THRESH:.3f}", (frame.shape[1] - 320, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Add blink state indicator
                state_names = ["OPEN", "CLOSING", "CLOSED", "OPENING"]
                cv2.putText(processed_frame, f"State: {state_names[self.blink_state]}", 
                           (frame.shape[1] - 320, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                
                # Add velocity display
                vel_color = (0, 0, 255) if self.debug_velocity < 0 else (0, 255, 0)
                cv2.putText(processed_frame, f"Velocity: {self.debug_velocity:.1f}", 
                          (frame.shape[1] - 320, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vel_color, 2)
            
            # Draw blink count and rate
            cv2.putText(processed_frame, f"Blinks: {self.total_blinks}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed_frame, f"Blink rate: {self.blink_rate:.1f} bpm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw average blink duration if we have data
            if self.blink_duration_history:
                cv2.putText(processed_frame, f"Avg duration: {self.avg_blink_duration*1000:.0f} ms", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw gaze direction
            gaze_color = (0, 0, 255) if self.direction_warning else (255, 0, 0)
            cv2.putText(processed_frame, f"Gaze: {current_direction}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
            
            # Draw warning indicators
            if self.direction_warning:
                cv2.putText(processed_frame, "WARNING: Looking " + current_direction + " too long!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if self.blink_rate_warning:
                if self.blink_rate < self.BLINK_RATE_LOW:
                    message = f"WARNING: Blink rate too low! ({self.blink_rate:.1f} < {self.BLINK_RATE_LOW})"
                else:
                    message = f"WARNING: Blink rate too high! ({self.blink_rate:.1f} > {self.BLINK_RATE_HIGH})"
                cv2.putText(processed_frame, message, (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No face detected
            cv2.putText(processed_frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return processed_frame, metrics
    
    def get_statistics(self):
        """Return summary statistics"""
        return {
            "total_blinks": self.total_blinks,
            "blink_rate": self.blink_rate,
            "elapsed_time": time.time() - self.start_time,
            "abnormal_blink_rate": self.blink_rate_warning,
            "gaze_direction": self.gaze_direction,
            "direction_warning": self.direction_warning,
            "calibrated": self.calibrated,
            "baseline_ear": self.baseline_ear,
            "threshold": self.EYE_AR_THRESH,
            "avg_blink_duration": self.avg_blink_duration if self.blink_duration_history else None
        }