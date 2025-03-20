import cv2
import dlib
import numpy as np
from scipy.spatial import distance

class GazeTracker:
    def __init__(self, shape_predictor_path):
        """
        Initialize the gaze tracker with a facial landmark predictor
        
        Args:
            shape_predictor_path: Path to dlib's facial landmark predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        
        # Parameters for calibration
        self.calibration_complete = False
        self.center_points = []
        self.calibration_frames = 30
        self.gaze_history = []
        self.history_size = 5  # Number of frames to use for smoothing
    
    def eye_aspect_ratio(self, eye_points):
        """
        Calculate the eye aspect ratio (EAR) to detect blinks
        
        Args:
            eye_points: 6 landmark points of an eye
            
        Returns:
            float: The eye aspect ratio
        """
        # Compute vertical distances
        v1 = distance.euclidean(eye_points[1], eye_points[5])
        v2 = distance.euclidean(eye_points[2], eye_points[4])
        
        # Compute horizontal distance
        h = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def extract_eye_regions(self, frame, landmarks, eye_indices):
        """
        Extract the eye region based on eye landmarks
        
        Args:
            frame: Input video frame
            landmarks: Facial landmarks
            eye_indices: Indices for the eye landmarks
            
        Returns:
            eye_region, eye_points, mask: Extracted eye region, landmark points, and eye mask
        """
        eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in eye_indices])
        
        # Create a tighter bounding rectangle
        x, y, w, h = cv2.boundingRect(eye_points)
        
        # Add some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        # Ensure we don't go beyond frame boundaries
        h, w_frame = frame.shape[:2]
        x_end = min(x + w, w_frame)
        y_end = min(y + h, h)
        
        eye_region = frame[y:y_end, x:x_end].copy()
        
        # Create a mask for the eye region
        mask = np.zeros((h, w), dtype=np.uint8)
        # Adjust points to be relative to the eye region
        relative_points = [(point[0] - x, point[1] - y) for point in eye_points]
        hull = cv2.convexHull(np.array(relative_points))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Ensure mask dimensions match eye_region
        mask = mask[:eye_region.shape[0], :eye_region.shape[1]]
        
        return eye_region, eye_points, (x, y, w, h), mask
    
    def detect_pupil(self, eye_region, mask=None):
        """
        Detect pupil in an eye region using advanced methods
        
        Args:
            eye_region: Image of the eye region
            mask: Optional mask to isolate the eye area
            
        Returns:
            center, bbox: Pupil center coordinates and bounding box
        """
        if eye_region.size == 0:
            return None, None
            
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply mask if provided
        if mask is not None:
            gray_eye = cv2.bitwise_and(gray_eye, gray_eye, mask=mask)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eye = clahe.apply(gray_eye)
        
        # Use adaptive thresholding for better pupil detection
        blurred_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        _, threshold_eye = cv2.threshold(blurred_eye, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Remove noise and enhance pupil region
        kernel = np.ones((3, 3), np.uint8)
        threshold_eye = cv2.erode(threshold_eye, kernel, iterations=2)
        threshold_eye = cv2.dilate(threshold_eye, kernel, iterations=1)
        
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20 or area > 1500:  # Adjust area thresholds
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.65:  # Increased circularity threshold
                    valid_contours.append((contour, area, circularity))
            
            if valid_contours:
                # Sort by area and circularity
                sorted_contours = sorted(valid_contours, key=lambda x: x[1] * x[2], reverse=True)
                pupil_contour = sorted_contours[0][0]
                
                # Get the center using moments for more accuracy
                M = cv2.moments(pupil_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    px, py, pw, ph = cv2.boundingRect(pupil_contour)
                    return (cx, cy), (px, py, pw, ph)
        
        return None, None

    def calibrate(self, pupil_data):
        """
        Calibrate the gaze tracker by collecting center looking data
        
        Args:
            pupil_data: Data about pupil positions
            
        Returns:
            bool: True if calibration is complete
        """
        if not self.calibration_complete and pupil_data["left_pupil"] and pupil_data["right_pupil"]:
            self.center_points.append((
                pupil_data["left_pupil"],
                pupil_data["right_pupil"],
                pupil_data["left_eye_rect"],
                pupil_data["right_eye_rect"]
            ))
            
            if len(self.center_points) >= self.calibration_frames:
                # Calculate average pupil positions when looking center
                left_pupils = [p[0] for p in self.center_points]
                right_pupils = [p[1] for p in self.center_points]
                
                self.center_left_pupil = np.mean(np.array(left_pupils), axis=0)
                self.center_right_pupil = np.mean(np.array(right_pupils), axis=0)
                
                # Calculate standard deviation for thresholds
                self.left_std = np.std(np.array(left_pupils), axis=0)
                self.right_std = np.std(np.array(right_pupils), axis=0)
                
                self.calibration_complete = True
                print("Calibration complete!")
                return True
        
        return False

    def determine_gaze_direction(self, left_pupil, right_pupil, left_eye_rect, right_eye_rect, ear_left, ear_right):
        """
        Determine the gaze direction based on pupil positions with improved thresholds
        """
        if ear_left < 0.2 and ear_right < 0.2:
            return "Eyes Closed"
            
        if not self.calibration_complete:
            return "Calibrating..."
        
        if not left_pupil or not right_pupil:
            return "Unable to track gaze"
        
        # Normalize pupil positions relative to eye width and height
        left_eye_width, left_eye_height = left_eye_rect[2], left_eye_rect[3]
        right_eye_width, right_eye_height = right_eye_rect[2], right_eye_rect[3]
        
        # Calculate relative positions from the center calibration
        left_x_rel = (left_pupil[0] - self.center_left_pupil[0]) / left_eye_width
        left_y_rel = (left_pupil[1] - self.center_left_pupil[1]) / left_eye_height
        
        right_x_rel = (right_pupil[0] - self.center_right_pupil[0]) / right_eye_width
        right_y_rel = (right_pupil[1] - self.center_right_pupil[1]) / right_eye_height
        
        # Take average of both eyes for final determination
        x_rel = (left_x_rel + right_x_rel) / 2
        y_rel = (left_y_rel + right_y_rel) / 2
        
        # Add to history for smoothing
        self.gaze_history.append((x_rel, y_rel))
        if len(self.gaze_history) > self.history_size:
            self.gaze_history.pop(0)
        
        # Use moving average for smoother results
        x_avg = np.mean([g[0] for g in self.gaze_history])
        y_avg = np.mean([g[1] for g in self.gaze_history])
        
        # Adjusted thresholds for more sensitive detection
        x_threshold = 0.08  # Reduced from 0.12
        y_threshold = 0.08  # Reduced from 0.12
        
        # Determine gaze direction with more granular thresholds
        if x_avg < -x_threshold and abs(y_avg) < y_threshold/2:
            return "Looking Left"
        elif x_avg > x_threshold and abs(y_avg) < y_threshold/2:
            return "Looking Right"
        elif y_avg < -y_threshold and abs(x_avg) < x_threshold/2:
            return "Looking Up"
        elif y_avg > y_threshold and abs(x_avg) < x_threshold/2:
            return "Looking Down"
        elif x_avg < -x_threshold and y_avg < -y_threshold:
            return "Looking Up-Left"
        elif x_avg > x_threshold and y_avg < -y_threshold:
            return "Looking Up-Right"
        elif x_avg < -x_threshold and y_avg > y_threshold:
            return "Looking Down-Left"
        elif x_avg > x_threshold and y_avg > y_threshold:
            return "Looking Down-Right"
        else:
            return "Looking Center"

    def process_frame(self, frame):
        """
        Process video frame to track gaze
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame, gaze_direction, pupil_data: Frame with annotations and detected gaze direction
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        processed_frame = frame.copy()
        
        pupil_data = {
            "left_pupil": None,
            "right_pupil": None,
            "left_eye_rect": None,
            "right_eye_rect": None
        }
        
        gaze_direction = "No face detected"
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Extract eye landmarks
            left_eye_landmarks = range(36, 42)
            right_eye_landmarks = range(42, 48)
            
            # Extract eye regions with masks
            left_eye_region, left_eye_points, left_eye_rect, left_eye_mask = self.extract_eye_regions(
                frame, landmarks, left_eye_landmarks)
            right_eye_region, right_eye_points, right_eye_rect, right_eye_mask = self.extract_eye_regions(
                frame, landmarks, right_eye_landmarks)
            
            # Calculate eye aspect ratio for blink detection
            ear_left = self.eye_aspect_ratio(
                [(landmarks.part(n).x, landmarks.part(n).y) for n in left_eye_landmarks])
            ear_right = self.eye_aspect_ratio(
                [(landmarks.part(n).x, landmarks.part(n).y) for n in right_eye_landmarks])
            
            # Detect pupils using masks
            left_pupil, left_bbox = self.detect_pupil(left_eye_region, left_eye_mask)
            right_pupil, right_bbox = self.detect_pupil(right_eye_region, right_eye_mask)
            
            pupil_data["left_pupil"] = left_pupil
            pupil_data["right_pupil"] = right_pupil
            pupil_data["left_eye_rect"] = left_eye_rect
            pupil_data["right_eye_rect"] = right_eye_rect
            
            # Draw bounding boxes for eyes
            cv2.rectangle(processed_frame, 
                          (left_eye_rect[0], left_eye_rect[1]), 
                          (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), 
                          (0, 255, 0), 2)
            cv2.rectangle(processed_frame, 
                          (right_eye_rect[0], right_eye_rect[1]), 
                          (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), 
                          (0, 255, 0), 2)
            
            # Draw pupils if detected
            if left_pupil:
                cv2.circle(processed_frame, 
                           (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 
                           3, (0, 0, 255), -1)
            
            if right_pupil:
                cv2.circle(processed_frame, 
                           (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 
                           3, (0, 0, 255), -1)
            
            # Check if calibration is in progress
            if not self.calibration_complete:
                self.calibrate(pupil_data)
                gaze_direction = f"Calibrating... {len(self.center_points)}/{self.calibration_frames}"
            else:
                # Determine gaze direction
                gaze_direction = self.determine_gaze_direction(
                    left_pupil, right_pupil, left_eye_rect, right_eye_rect, ear_left, ear_right)
                
            # Display gaze direction on frame
            cv2.putText(processed_frame, f"Gaze: {gaze_direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display EAR values
            cv2.putText(processed_frame, f"EAR: {(ear_left + ear_right) / 2:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return processed_frame, gaze_direction, pupil_data

# Example usage:
def main():
    tracker = GazeTracker("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)
    
    print("Calibration: Look at the center of the screen for a few seconds...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, gaze_direction, _ = tracker.process_frame(frame)
        
        cv2.putText(processed_frame, "Press 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Gaze Tracking", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()