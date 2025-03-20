import cv2
import numpy as np
import time

class MultipleFaceDetector:
    """
    A dedicated module for detecting multiple faces in a frame using Haar cascades or DNN.
    """
    def __init__(self, method='cascade', 
                 model_path=None, 
                 config_path=None,
                 confidence_threshold=0.5,
                 scale_factor=1.1,
                 min_neighbors=5,
                 min_size=(30, 30)):
        """
        Initialize the face detector with the chosen method.
        
        Args:
            method (str): Detection method, either 'cascade' (faster) or 'dnn' (more accurate)
            model_path (str): Path to face detection model
            config_path (str): Path to model config (for DNN only)
            confidence_threshold (float): Confidence threshold for DNN detection
            scale_factor (float): Scale factor for cascade detection
            min_neighbors (int): Minimum neighbors for cascade detection
            min_size (tuple): Minimum face size for detection
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Statistics tracking
        self.total_frames = 0
        self.total_faces_detected = 0
        self.max_faces_detected = 0
        self.face_detection_start_time = time.time()
        self.face_counts = {}  # Dictionary to track frequency of different face counts
        
        # Initialize the appropriate detector based on method
        if method == 'cascade':
            # Use default cascade if none specified
            if model_path is None:
                model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.detector = cv2.CascadeClassifier(model_path)
            if self.detector.empty():
                raise ValueError(f"Could not load face cascade classifier from {model_path}")
                
        elif method == 'dnn':
            # Use Caffe-based DNN face detector (more accurate but slower)
            if model_path is None or config_path is None:
                raise ValueError("DNN method requires both model_path and config_path")
                
            self.detector = cv2.dnn.readNetFromCaffe(config_path, model_path)
            
        else:
            raise ValueError(f"Unknown face detection method: {method}")
            
        print(f"MultipleFaceDetector initialized using {method} method")
    
    def detect_faces(self, frame):
        """
        Detect all faces in the frame.
        
        Args:
            frame: The input video frame
            
        Returns:
            tuple: (frame with annotations, face_data)
                - frame with annotations: The input frame with detected faces marked
                - face_data: Dictionary with detection information
        """
        self.total_frames += 1
        frame_height, frame_width = frame.shape[:2]
        faces = []
        
        if self.method == 'cascade':
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            detections = self.detector.detectMultiScale(
                gray, 
                scaleFactor=self.scale_factor, 
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            # Process detected faces
            for (x, y, w, h) in detections:
                faces.append({
                    'box': (x, y, w, h),
                    'confidence': 1.0  # Cascade doesn't provide confidence scores
                })
                
        elif self.method == 'dnn':
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Pass the blob through the network
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            # Process each detection
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence > self.confidence_threshold:
                    # Compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure the bounding box falls within the dimensions of the frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(frame_width - 1, endX)
                    endY = min(frame_height - 1, endY)
                    
                    # Extract the face ROI
                    w = endX - startX
                    h = endY - startY
                    
                    faces.append({
                        'box': (startX, startY, w, h),
                        'confidence': float(confidence)
                    })
                    
        # Update statistics
        num_faces = len(faces)
        self.total_faces_detected += num_faces
        self.max_faces_detected = max(self.max_faces_detected, num_faces)
        
        # Update face count frequency
        face_count_key = str(num_faces)
        if face_count_key in self.face_counts:
            self.face_counts[face_count_key] += 1
        else:
            self.face_counts[face_count_key] = 1
            
        # Create a annotated copy of the frame
        annotated_frame = frame.copy()
        
        # Draw face markers with alternating colors for better visibility
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            color = colors[i % len(colors)]
            confidence = face['confidence']
            
            # Draw rectangle around face
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            
            # Add confidence score label
            label = f"Face {i+1}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Add a warning overlay for multiple faces
        if num_faces > 1:
            # Create a semi-transparent overlay
            overlay = annotated_frame.copy()
            rect_width = 400
            rect_height = 60
            rect_x = (frame_width - rect_width) // 2
            rect_y = (frame_height - rect_height) // 2
            cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
            
            # Add the warning text
            cv2.putText(overlay, f"MULTIPLE FACES DETECTED: {num_faces}", 
                       (rect_x + 20, rect_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Apply the overlay with transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
        # Package detection data
        face_data = {
            'face_count': num_faces,
            'faces': faces,
            'multiple_faces_detected': num_faces > 1,
            'face_coordinates': [face['box'] for face in faces]
        }
        
        return annotated_frame, face_data
        
    def get_statistics(self):
        """
        Get statistics on face detection.
        
        Returns:
            dict: Statistics dictionary
        """
        elapsed_time = time.time() - self.face_detection_start_time
        
        stats = {
            'total_frames': self.total_frames,
            'total_faces_detected': self.total_faces_detected,
            'average_faces_per_frame': self.total_faces_detected / max(1, self.total_frames),
            'max_faces_detected': self.max_faces_detected,
            'elapsed_time': elapsed_time,
            'face_count_distribution': self.face_counts
        }
        
        return stats