import cv2
import time
import numpy as np
from ultralytics import YOLO

class MaskDetector:
    def __init__(self, model_path='model/mask_detection.pt', confidence_threshold=0.5):
        """
        Initialize the mask detection module.
        
        Args:
            model_path (str): Path to the YOLOv12 mask detection model weights
            confidence_threshold (float): Confidence threshold for detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        
        # Statistics
        self.total_frames = 0
        self.frames_without_mask = 0
        self.last_mask_status = None
        self.detection_time = time.time()
        
        print(f"Loaded mask detection model from {model_path}")
    
    def detect_mask(self, frame):
        """
        Detect if a person is wearing a mask in the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (processed_frame, mask_metrics)
        """
        # Make a copy of the frame
        processed_frame = frame.copy()
        self.total_frames += 1
        
        # Run YOLOv12 inference on the frame
        results = self.model(frame)
        
        # Process the results
        mask_detected = False
        no_mask_detected = False
        
        mask_metrics = {
            "mask_status": "No face detected",
            "mask_detected": False,
            "no_mask_detected": False,
            "confidence": 0.0,
            "detection_time": 0.0
        }
        
        # Check if we have any detections
        if results[0].boxes.data.numel() > 0:
            for box in results[0].boxes.data:
                # Extract box information
                x1, y1, x2, y2, conf, cls = box
                
                # Skip if confidence is below threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Get class information
                class_id = int(cls.item())
                class_name = self.model.names[class_id]
                
                # Determine color based on class (mask or no-mask)
                if "mask" in class_name.lower() and "no" not in class_name.lower():
                    color = (0, 255, 0)  # Green for mask
                    mask_detected = True
                    status = "Mask Detected"
                else:
                    color = (0, 0, 255)  # Red for no-mask
                    no_mask_detected = True
                    status = "No Mask"
                
                # Draw bounding box
                cv2.rectangle(
                    processed_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )
                
                # Add label with confidence
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(
                    processed_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                
                # Update mask metrics
                mask_metrics["confidence"] = conf.item()
                
        # Update mask status and statistics
        if mask_detected and not no_mask_detected:
            mask_metrics["mask_status"] = "Mask Detected"
            mask_metrics["mask_detected"] = True
            self.last_mask_status = True
        elif no_mask_detected:
            mask_metrics["mask_status"] = "No Mask"
            mask_metrics["no_mask_detected"] = True
            self.last_mask_status = False
            self.frames_without_mask += 1
        else:
            if self.last_mask_status is not None:
                mask_metrics["mask_status"] = "Mask Detected" if self.last_mask_status else "No Mask"
                mask_metrics["mask_detected"] = self.last_mask_status
                mask_metrics["no_mask_detected"] = not self.last_mask_status
        
        # Update detection time
        current_time = time.time()
        mask_metrics["detection_time"] = current_time - self.detection_time
        self.detection_time = current_time
        
        # Add current status to frame
        status_color = (0, 255, 0) if mask_metrics["mask_detected"] else (0, 0, 255)
        cv2.putText(
            processed_frame,
            f"Status: {mask_metrics['mask_status']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )
        
        return processed_frame, mask_metrics
    
    def get_statistics(self):
        """
        Get mask detection statistics.
        
        Returns:
            dict: Statistics about mask detection
        """
        stats = {
            "total_frames": self.total_frames,
            "frames_without_mask": self.frames_without_mask,
            "mask_compliance_rate": 0.0
        }
        
        if self.total_frames > 0:
            stats["mask_compliance_rate"] = (self.total_frames - self.frames_without_mask) / self.total_frames * 100
        
        return stats