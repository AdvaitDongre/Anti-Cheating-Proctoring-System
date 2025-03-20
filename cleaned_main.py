import cv2
import time
import argparse
import numpy as np
from eye_monitor import EyeMonitor
from cheating_detector import CheatingDetector
from head_posture_monitor import process_head_pose
from pupil_tracker import PupilTracker
from mask_detector import MaskDetector
from multiple_face_detector import MultipleFaceDetector

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Clean eye monitoring and cheating detection system')
    parser.add_argument('--predictor', default='model/shape_predictor_68_face_landmarks.dat',
                        help='Path to the facial landmark predictor')
    parser.add_argument('--yolo-model', default='model/yolo12n.pt',
                        help='Path to the YOLOv12 model weights')
    parser.add_argument('--video-source', type=int, default=0,
                        help='Video source (webcam index, default 0)')
    parser.add_argument('--video-output', type=str, default=None,
                        help='Output video file path (optional)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output and visualization')
    
    args = parser.parse_args()
    
    # Initialize monitors
    try:
        eye_monitor = EyeMonitor(
            shape_predictor_path=args.predictor,
            blink_threshold=0.25,
            blink_consec_frames=2,
            gaze_threshold=0.3,
            direction_time_threshold=3.0,
            blink_rate_low=10,
            blink_rate_high=30
        )
        cheating_detector = CheatingDetector(
            model_path=args.yolo_model,
            confidence_threshold=0.5
        )
        face_detector = MultipleFaceDetector(
            method='cascade',
            confidence_threshold=0.5
        )
        print("System initialized successfully.")
    except Exception as e:
        print(f"Error initializing system: {e}")
        return
    
    # Start video capture
    cap = cv2.VideoCapture(args.video_source)
    if not cap.isOpened():
        print(f"Could not open video source {args.video_source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    video_writer = None
    if args.video_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.video_output, fourcc, fps_cap, (frame_width, frame_height)
        )
    
    print("Monitoring system started. Press 'q' to exit.")
    
    # Initialize time for FPS calculation
    fps_start_time = time.time()
    frame_count = 0
    fps = 0
    
    # Initialize log time to avoid spamming console
    last_log_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame with eye monitor
        eye_frame, eye_metrics = eye_monitor.process_frame(frame)
        
        # Process frame with cheating detector
        cheat_frame, cheat_metrics = cheating_detector.detect_objects(frame)
        
        # Process frame with face detector
        face_frame, face_data = face_detector.detect_faces(frame)
        
        # Create clean display frame
        display_frame = frame.copy()
        
        # Add essential information overlay
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Add FPS counter
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, y_offset),
                   font, font_scale, (255, 255, 255), thickness)
        y_offset += 30
        
        # Add eye metrics if face is detected
        if eye_metrics['face_detected']:
            # Show blink count and rate
            cv2.putText(display_frame, f"Blinks: {eye_metrics['blinks']}", 
                       (10, y_offset), font, font_scale, (255, 255, 255), thickness)
            y_offset += 30
            
            # Show EAR if calibrated
            if eye_metrics.get('calibrated', False):
                ear_value = eye_metrics.get('ear', 0)
                ear_color = (0, 0, 255) if ear_value < eye_monitor.EYE_AR_THRESH else (0, 255, 0)
                cv2.putText(display_frame, f"EAR: {ear_value:.4f}", 
                           (10, y_offset), font, font_scale, ear_color, thickness)
                y_offset += 30
            
            # Show gaze direction only if not looking at screen
            if eye_metrics['gaze_direction'] != "Looking at Screen":
                cv2.putText(display_frame, f"Gaze: {eye_metrics['gaze_direction']}", 
                           (10, y_offset), font, font_scale, (0, 0, 255), thickness)
                y_offset += 30
        
        # Add face count information
        if face_data['face_count'] > 0:
            face_color = (0, 0, 255) if face_data['multiple_faces_detected'] else (0, 255, 0)
            cv2.putText(display_frame, f"Faces: {face_data['face_count']}", 
                       (10, y_offset), font, font_scale, face_color, thickness)
            y_offset += 30
        
        # Add cheating detection warning
        if cheat_metrics['cheating_detected']:
            cv2.putText(display_frame, "⚠️ Cheating Detected!", 
                       (10, y_offset), font, font_scale, (0, 0, 255), thickness)
            y_offset += 30
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        
        # Log to console only every 1 second
        if time.time() - last_log_time > 1.0:
            last_log_time = time.time()
            log_str = f"FPS: {fps:.1f} | Blinks: {eye_metrics['blinks']}"
            if eye_metrics['face_detected']:
                log_str += f" | EAR: {eye_metrics.get('ear', 0):.4f}"
            if face_data['face_count'] > 0:
                log_str += f" | Faces: {face_data['face_count']}"
            if cheat_metrics['cheating_detected']:
                log_str += " | ⚠️ Cheating Detected!"
            print(log_str)
        
        # Display the frame
        cv2.imshow("Clean Monitoring System", display_frame)
        
        # Save to video file if specified
        if video_writer:
            video_writer.write(display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.close()
    cv2.destroyAllWindows()
    
    # Print summary statistics
    print("\nMonitoring session summary:")
    stats = eye_monitor.get_statistics()
    print(f"- Total monitoring time: {stats['elapsed_time']:.1f} seconds")
    print(f"- Total blinks detected: {stats['total_blinks']}")
    print(f"- Average blink rate: {stats['blink_rate']:.1f} blinks per minute")
    
    # Get cheating detection summary
    cheat_stats = cheating_detector.get_statistics()
    print(f"- Cheating instances detected: {cheat_stats['total_cheating_instances']}")
    
    # Show face detection statistics
    face_stats = face_detector.get_statistics()
    print(f"- Total faces detected: {face_stats['total_faces_detected']}")
    print(f"- Maximum faces detected: {face_stats['max_faces_detected']}")

if __name__ == "__main__":
    main() 