import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path

class YOLOv12MultiPersonDetector:
    def __init__(self, model_path='yolov12.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize YOLOv12 for multi-person detection.
        
        Args:
            model_path (str): Path to YOLOv12 model weights (.pt file)
            conf_threshold (float): Confidence threshold (0-1)
            iou_threshold (float): IoU threshold for NMS (0-1)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = 0  # COCO 'person' class ID

    def detect(self, frame):
        """Detect people in a frame and return annotated frame + count."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False
        )
        annotated_frame = results[0].plot()
        person_count = len(results[0].boxes)
        
        # Add warning if more than 2 people detected
        if person_count >= 2:
            warning_msg = "WARNING: More than 1 person detected!"
            print(warning_msg)
            cv2.putText(annotated_frame, warning_msg, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return annotated_frame, person_count

    def process_image(self, image_path, output_path='output.jpg'):
        """Detect people in a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return

        annotated_frame, person_count = self.detect(frame)
        cv2.putText(annotated_frame, f'People: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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

            annotated_frame, person_count = self.detect(frame)
            cv2.putText(annotated_frame, f'People: {person_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
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

            annotated_frame, person_count = self.detect(frame)
            cv2.putText(annotated_frame, f'People: {person_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Webcam Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Multi-Person Detection")
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'video', 'webcam'], help="Detection mode")
    parser.add_argument('--source', type=str, default=None, help="Path to input file (image/video)")
    parser.add_argument('--output', type=str, default=None, help="Path to save output (image/video)")
    parser.add_argument('--model', type=str, default='../model/yolo12n.pt', help="Path to YOLOv12 model")
    args = parser.parse_args()

    detector = YOLOv12MultiPersonDetector(model_path=args.model)

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