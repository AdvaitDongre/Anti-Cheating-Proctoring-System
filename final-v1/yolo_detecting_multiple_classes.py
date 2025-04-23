import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path

class YOLOv12ExamCheatingDetector:
    def __init__(self, model_path='yolov12.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize YOLOv12 for exam cheating detection.
        
        Args:
            model_path (str): Path to YOLOv12 model weights (.pt file)
            conf_threshold (float): Confidence threshold (0-1)
            iou_threshold (float): IoU threshold for NMS (0-1)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO class IDs for cheating-related objects
        self.class_ids = {
            'person': 0,
            'cell phone': 67,
            'laptop': 63,
            'book': 73,
            'mouse': 74,
            'keyboard': 66,
            'tv': 62,
            'handbag': 26,
            'backpack': 24,
            'paper': 85,
            'clock': 74  # could be used to detect looking at watch frequently
        }

    def detect(self, frame):
        """Detect cheating-related objects in a frame and return annotated frame + counts."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=list(self.class_ids.values()),
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        detections = results[0].boxes
        
        # Count each class of interest
        counts = {class_name: 0 for class_name in self.class_ids}
        for detection in detections:
            cls = int(detection.cls)
            for class_name, class_id in self.class_ids.items():
                if cls == class_id:
                    counts[class_name] += 1
        
        # Add warnings for suspicious items
        warnings = []
        if counts['cell phone'] > 0:
            warnings.append("Cell phone detected!")
        if counts['laptop'] > 0:
            warnings.append("Laptop detected!")
        if counts['person'] > 1:
            warnings.append("Multiple people detected!")
        if counts['mouse'] > 0 or counts['keyboard'] > 0:
            warnings.append("Computer peripherals detected!")
            
        # Display warnings
        for i, warning in enumerate(warnings):
            cv2.putText(annotated_frame, warning, (10, frame.shape[0] - 30 - (i*30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(warning)
            
        return annotated_frame, counts

    def process_image(self, image_path, output_path='output.jpg'):
        """Detect people in a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return

        annotated_frame, _ = self.detect(frame)  # Ignore counts in return
        
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved result to {output_path}")
        cv2.imshow("Detection Result", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path, output_path='output.mp4'):
        """Process a video file and save results."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        print(f"Processing video... (Press 'q' to stop early)")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, _ = self.detect(frame)  # Ignore counts in return
            
            out.write(annotated_frame)
            cv2.imshow("Video Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved processed video to {output_path}")

    def process_webcam(self):
        """Real-time detection from webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Webcam active. Press 'q' to quit...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            annotated_frame, _ = self.detect(frame)  # Ignore counts in return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Webcam Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Multi-Person Detection")
    parser.add_argument('--mode', type=str, default='webcam', choices=['image', 'video', 'webcam'], help="Detection mode")
    parser.add_argument('--source', type=str, default=None, help="Path to input file (image/video)")
    parser.add_argument('--output', type=str, default=None, help="Path to save output (image/video)")
    parser.add_argument('--model', type=str, default='../model/yolo12n.pt', help="Path to YOLOv12 model")
    args = parser.parse_args()

    detector = YOLOv12ExamCheatingDetector(model_path=args.model)

    if args.mode == 'image':
        if not args.source:
            raise ValueError("--source required for image mode")
        output_path = args.output if args.output else 'output.jpg'
        detector.process_image(args.source, output_path)
    
    elif args.mode == 'video':
        if not args.source:
            raise ValueError("--source required for video mode")
        output_path = args.output if args.output else 'output.mp4'
        detector.process_video(args.source, output_path)
    
    elif args.mode == 'webcam':
        detector.process_webcam()