import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

class CheatingDetector:
    def __init__(
        self,
        model_path="model/mobile.pt",  # Use the nano version by default
        confidence_threshold=0.5,
        nms_threshold=0.45,
        image_size=640,
    ):
        """
        Initialize the cheating detector with YOLOv12 model using Ultralytics
        
        Args:
            model_path: Path to the YOLOv12 model weights
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            image_size: Input image size for the model
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.image_size = image_size
        
        # Load YOLOv12 model using Ultralytics
        try:
            self.model = YOLO(model_path)
            
            # Target classes we're interested in
            self.target_classes = [
                'cell phone', 'laptop', 'book', 'remote', 'keyboard', 
                'mouse', 'tablet', 'cup', 'bottle', 'scissors', 
                'person', 'tie', 'backpack', 'handbag'
            ]
            
            print(f"CheatingDetector initialized using Ultralytics YOLOv12")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv12 model: {e}")
        
        # Initialize metrics
        self.cheating_detected = False
        self.cheating_start_time = None
        self.total_cheating_time = 0
        self.cheating_instances = 0
        self.detected_objects = {}
        self.start_time = time.time()
    
    def detect_objects(self, frame):
        """
        Detect objects in the frame that could indicate cheating
        
        Args:
            frame: Input frame from the camera
        
        Returns:
            processed_frame: Frame with detection boxes
            detections: Dictionary containing detection results
        """
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Perform inference with Ultralytics YOLO
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.image_size,
            verbose=False
        )[0]
        
        # Process results
        detections = {
            'objects_detected': [],
            'cheating_detected': False,
            'cheating_objects': [],
            'frame_with_boxes': annotated_frame
        }
        
        # Check if any detections were made
        if results.boxes:
            # Extract detection results
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = results.names[cls]
                
                # Check if detected object is in our target classes
                if cls_name in self.target_classes:
                    # Add to detected objects
                    detections['objects_detected'].append({
                        'class': cls_name,
                        'confidence': conf,
                        'box': (x1, y1, x2, y2)
                    })
                    
                    # Check if this object indicates cheating
                    if cls_name in ['cell phone', 'laptop', 'book', 'remote', 'keyboard', 
                                   'mouse', 'tablet', 'backpack', 'handbag']:
                        detections['cheating_detected'] = True
                        detections['cheating_objects'].append(cls_name)
                        
                        # Draw red box for cheating objects
                        cv2.rectangle(detections['frame_with_boxes'], (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{cls_name}: {conf:.2f}"
                        cv2.putText(detections['frame_with_boxes'], label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # Draw green box for non-cheating objects
                        cv2.rectangle(detections['frame_with_boxes'], (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cls_name}: {conf:.2f}"
                        cv2.putText(detections['frame_with_boxes'], label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update cheating metrics
        self._update_metrics(detections['cheating_detected'])
        
        # Add warning text if cheating detected
        if detections['cheating_detected']:
            cv2.putText(detections['frame_with_boxes'], "CHEATING DETECTED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            objects_text = f"Objects: {', '.join(detections['cheating_objects'])}"
            cv2.putText(detections['frame_with_boxes'], objects_text, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return detections['frame_with_boxes'], detections
    
    def _update_metrics(self, cheating_detected):
        """Update cheating metrics based on current detection"""
        current_time = time.time()
        
        # Update cheating state
        if cheating_detected and not self.cheating_detected:
            # Cheating just started
            self.cheating_detected = True
            self.cheating_start_time = current_time
            self.cheating_instances += 1
        elif not cheating_detected and self.cheating_detected:
            # Cheating just ended
            self.cheating_detected = False
            if self.cheating_start_time is not None:
                self.total_cheating_time += (current_time - self.cheating_start_time)
    
    def get_statistics(self):
        """Return statistics about cheating detection"""
        elapsed_time = time.time() - self.start_time
        
        # If currently cheating, add the current session
        current_cheating_time = self.total_cheating_time
        if self.cheating_detected and self.cheating_start_time is not None:
            current_cheating_time += (time.time() - self.cheating_start_time)
        
        return {
            'total_cheating_instances': self.cheating_instances,
            'total_cheating_time': current_cheating_time,
            'cheating_percentage': current_cheating_time / elapsed_time * 100 if elapsed_time > 0 else 0,
            'elapsed_time': elapsed_time,
            'currently_cheating': self.cheating_detected
        }
    
    def detect_face_mask(self, face_img):
        """
        Specialized method to detect if someone is wearing a mask
        
        Note: This would work better with a model specifically trained for mask detection
        """
        # Perform inference directly on the face image
        results = self.model.predict(
            source=face_img,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.image_size,
            verbose=False
        )[0]
        
        # For mask detection, you'd need to either:
        # 1. Fine-tune YOLOv12 on mask data
        # 2. Use a specialized mask detection model
        
        # This is a placeholder approach that checks for objects that might cover the face
        for box in results.boxes:
            cls = int(box.cls[0])
            cls_name = results.names[cls]
            conf = float(box.conf[0])
            
            # In a real implementation, you'd have trained the model to detect different types of masks
            if cls_name in ['tie', 'cell phone'] and conf > 0.5:
                # These objects might be covering the face in some way
                return True
                
        return False