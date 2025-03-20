import cv2
import time
import argparse
import numpy as np
from eye_monitor import EyeMonitor
from cheating_detector import CheatingDetector
from head_posture_monitor import process_head_pose
from pupil_tracker import GazeTracker  # Update import to use GazeTracker instead of PupilTracker
from mask_detector import MaskDetector  # Import the mask detector
from multiple_face_detector import MultipleFaceDetector  # Import the multiple face detector

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Eye monitoring and cheating detection system')
    # Eye monitoring arguments
    parser.add_argument('--predictor', default='model/shape_predictor_68_face_landmarks.dat',
                        help='Path to the facial landmark predictor')
    parser.add_argument('--blink-threshold', type=float, default=0.25,
                        help='Initial eye aspect ratio threshold for blink detection (will be calibrated)')
    parser.add_argument('--blink-consec-frames', type=int, default=2,
                        help='Number of consecutive frames for blink confirmation')
    parser.add_argument('--gaze-threshold', type=float, default=0.3,
                        help='Threshold for determining gaze direction')
    parser.add_argument('--direction-time', type=float, default=3.0,
                        help='Time threshold (in seconds) for gaze direction warning')
    parser.add_argument('--blink-rate-low', type=float, default=10,
                        help='Lower threshold for normal blink rate (blinks per minute)')
    parser.add_argument('--blink-rate-high', type=float, default=30,
                        help='Upper threshold for normal blink rate (blinks per minute)')
    
    # Cheating detection arguments
    parser.add_argument('--yolo-model', default='model/yolo12n.pt',
                        help='Path to the YOLOv12 model weights')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for object detection')
    
    # Head posture monitoring arguments
    parser.add_argument('--head-calibration-time', type=float, default=5.0,
                        help='Time (in seconds) for head posture calibration')
    
    # Pupil tracking arguments
    parser.add_argument('--use-pupil-tracking', action='store_true',
                        help='Enable pupil tracking for more accurate gaze detection')
    parser.add_argument('--pupil-threshold', type=int, default=50,
                        help='Threshold for pupil detection')
    
    # Mask detection arguments
    parser.add_argument('--mask-model', default='model/mask_detection.pt',
                        help='Path to the YOLOv12 mask detection model weights')
    parser.add_argument('--mask-conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for mask detection')
    parser.add_argument('--use-mask-detection', action='store_true',
                        help='Enable mask detection')
    
    # Multiple face detection arguments
    parser.add_argument('--face-detection-method', choices=['cascade', 'dnn'], default='cascade',
                        help='Method for multiple face detection')
    parser.add_argument('--face-conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for face detection')
    
    # General arguments
    parser.add_argument('--display-mode', choices=['split', 'eye', 'cheat', 'head', 'pupil', 'mask', 'combined', 'faces'], 
                        default='combined', help='Display mode for visualization')
    parser.add_argument('--video-source', type=int, default=0,
                        help='Video source (webcam index, default 0)')
    parser.add_argument('--video-output', type=str, default=None,
                        help='Output video file path (optional)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output and visualization')
    
    args = parser.parse_args()
    
    # Initialize eye monitor
    try:
        eye_monitor = EyeMonitor(
            shape_predictor_path=args.predictor,
            blink_threshold=args.blink_threshold,
            blink_consec_frames=args.blink_consec_frames,
            gaze_threshold=args.gaze_threshold,
            direction_time_threshold=args.direction_time,
            blink_rate_low=args.blink_rate_low,
            blink_rate_high=args.blink_rate_high
        )
        print("Eye monitor initialized successfully.")
    except Exception as e:
        print(f"Error initializing EyeMonitor: {e}")
        print("Make sure the shape predictor file exists at the specified path.")
        return
    
    # Initialize cheating detector
    try:
        cheating_detector = CheatingDetector(
            model_path=args.yolo_model,
            confidence_threshold=args.conf_threshold
        )
        print("Cheating detector initialized successfully.")
    except Exception as e:
        print(f"Error initializing CheatingDetector: {e}")
        print("Make sure the YOLOv12 model file exists at the specified path.")
        return
    
    # Initialize pupil tracker if enabled
    pupil_tracker = None
    if args.use_pupil_tracking:
        try:
            pupil_tracker = GazeTracker(args.predictor)  # Update to use GazeTracker
            print("Pupil tracker initialized successfully.")
        except Exception as e:
            print(f"Error initializing GazeTracker: {e}")
            print("Pupil tracking will be disabled.")
    
    # Initialize mask detector if enabled
    mask_detector = None
    if args.use_mask_detection:
        try:
            mask_detector = MaskDetector(
                model_path=args.mask_model,
                confidence_threshold=args.mask_conf_threshold
            )
            print("Mask detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing MaskDetector: {e}")
            print("Mask detection will be disabled.")
    
    # Initialize multiple face detector
    try:
        face_detector = MultipleFaceDetector(
            method=args.face_detection_method,
            confidence_threshold=args.face_conf_threshold
        )
        print("Multiple face detector initialized successfully.")
    except Exception as e:
        print(f"Error initializing MultipleFaceDetector: {e}")
        print("Multiple face detection will be disabled.")
        face_detector = None
    
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
    
    print("Monitoring system started.")
    print("Calibration phase: Please look at the camera with eyes open for a few seconds.")
    print("Press 'q' to exit, 'm' to toggle display mode.")
    
    # Initialize time for FPS calculation
    fps_start_time = time.time()
    frame_count = 0
    fps = 0
    
    # Initialize display mode
    display_mode = args.display_mode
    
    # Initialize log time to avoid spamming console
    last_log_time = time.time()
    
    # Initialize head posture calibration
    head_calibration_start = time.time()
    head_calibration_frames = []
    calibrated_head_angles = None
    head_calibration_phase = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Make sure we always have a fresh copy of the frame for each processor
        original_frame = frame.copy()
        
        # Process frame with eye monitor
        eye_frame, eye_metrics = eye_monitor.process_frame(original_frame)
        
        # Process frame with cheating detector - use original frame, not the eye-processed one
        cheat_frame, cheat_metrics = cheating_detector.detect_objects(original_frame)
        
        # Process frame with head posture monitor
        head_frame = original_frame.copy()
        head_direction = "Calibrating..."
        
        # Process frame with pupil tracker if enabled
        pupil_frame = original_frame.copy()
        pupil_gaze_direction = "Unknown"
        pupil_data = None
        
        if pupil_tracker:
            pupil_frame, pupil_gaze_direction, pupil_data = pupil_tracker.process_frame(pupil_frame)  # Update method call
            
            # If we have pupil data and it's more accurate, use it to update eye metrics
            if pupil_data and pupil_data["left_pupil"] and pupil_data["right_pupil"]:
                eye_metrics['gaze_direction'] = pupil_gaze_direction
        
        # Process frame with mask detector if enabled
        mask_frame = original_frame.copy()
        mask_status = "Mask detection disabled"
        mask_metrics = {
            "mask_status": "Disabled",
            "mask_detected": False,
            "no_mask_detected": False
        }
        
        if mask_detector:
            mask_frame, mask_metrics = mask_detector.detect_mask(mask_frame)
            mask_status = mask_metrics["mask_status"]
        
        # Process frame with multiple face detector
        face_frame = original_frame.copy()
        face_data = {'face_count': 0, 'faces': [], 'multiple_faces_detected': False}
        if face_detector:
            face_frame, face_data = face_detector.detect_faces(face_frame)
        
        # Handle head posture calibration
        if head_calibration_phase:
            if time.time() - head_calibration_start < args.head_calibration_time:
                head_frame, angles = process_head_pose(head_frame)

                if angles is not None:
                    # Ensure angles are numeric before appending
                    if all(isinstance(a, (int, float)) for a in angles):
                        head_calibration_frames.append(angles)
                    else:
                        print("Invalid angle data type detected:", angles)

                head_direction = "Calibrating head posture..."

            else:
                if head_calibration_frames:
                    try:
                        # Ensure we convert to a float array before averaging
                        calibrated_angles = np.mean(np.array(head_calibration_frames, dtype=float), axis=0)
                        calibrated_head_angles = calibrated_angles
                        print(f"Head posture calibration complete. Calibrated angles: {calibrated_head_angles}")
                    except Exception as e:
                        print("Error during mean calculation:", e)
                else:
                    print("Head posture calibration failed. No face detected during calibration.")

                head_calibration_phase = False

        else:
            # Ensure calibrated_head_angles exists and is valid before use
            if 'calibrated_head_angles' in locals():
                head_frame, head_direction = process_head_pose(head_frame, calibrated_head_angles)
            else:
                print("Error: calibrated_head_angles not set.")
                head_frame, head_direction = process_head_pose(head_frame)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        
        # Create display frame based on mode
        if display_mode == 'split':
            # Create a split view with mask detection if enabled
            # Resize frames to make them equal size
            h, w = frame.shape[:2]
            
            # Determine grid layout based on which detectors are enabled
            grid_width = 2
            grid_height = 2
            
            if mask_detector:
                grid_width = 3  # Make it a 3x2 grid
            
            cell_width = w // grid_width
            cell_height = h // grid_height
            
            # Resize all frames
            eye_frame_resized = cv2.resize(eye_frame, (cell_width, cell_height))
            cheat_frame_resized = cv2.resize(cheat_frame, (cell_width, cell_height))
            head_frame_resized = cv2.resize(head_frame, (cell_width, cell_height))
            
            if pupil_tracker:
                pupil_frame_resized = cv2.resize(pupil_frame, (cell_width, cell_height))
            else:
                pupil_frame_resized = cv2.resize(original_frame, (cell_width, cell_height))
            
            if mask_detector:
                mask_frame_resized = cv2.resize(mask_frame, (cell_width, cell_height))
            
            # Add labels
            cv2.putText(eye_frame_resized, "Eye Monitor", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(cheat_frame_resized, "Cheating Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(head_frame_resized, "Head Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if pupil_tracker:
                cv2.putText(pupil_frame_resized, "Pupil Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(pupil_frame_resized, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if mask_detector:
                cv2.putText(mask_frame_resized, "Mask Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine frames based on which detectors are enabled
            if mask_detector:
                # Create a 3x2 grid
                top_row = cv2.hconcat([eye_frame_resized, cheat_frame_resized, head_frame_resized])
                bottom_row = cv2.hconcat([pupil_frame_resized, mask_frame_resized, original_frame.copy()])
            else:
                # Create a 2x2 grid
                top_row = cv2.hconcat([eye_frame_resized, cheat_frame_resized])
                bottom_row = cv2.hconcat([head_frame_resized, pupil_frame_resized])
            
            display_frame = cv2.vconcat([top_row, bottom_row])
            
        elif display_mode == 'eye':
            display_frame = eye_frame
        elif display_mode == 'cheat':
            display_frame = cheat_frame
        elif display_mode == 'head':
            display_frame = head_frame
        elif display_mode == 'pupil' and pupil_tracker:
            display_frame = pupil_frame
        elif display_mode == 'mask' and mask_detector:
            display_frame = mask_frame
        elif display_mode == 'faces' and face_detector:
            display_frame = face_frame
        else:  # combined mode
            # Begin with cheating detection frame as base
            display_frame = cheat_frame.copy()
            
            # Add eye metrics overlay to the top-left corner
            if eye_metrics['face_detected']:
                # Add bounding box around eye region if face detected
                if 'face_box' in eye_metrics:
                    x, y, w, h = eye_metrics['face_box']
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # Add pupil tracking visuals if enabled
            if pupil_tracker and pupil_data:
                # Draw pupil positions on the combined frame
                left_eye_rect = pupil_data["left_eye_rect"]
                right_eye_rect = pupil_data["right_eye_rect"]
                left_pupil = pupil_data["left_pupil"]
                right_pupil = pupil_data["right_pupil"]
                
                if left_eye_rect:
                    cv2.rectangle(display_frame, (left_eye_rect[0], left_eye_rect[1]), 
                                 (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), 
                                 (0, 255, 0), 1)
                    
                if right_eye_rect:
                    cv2.rectangle(display_frame, (right_eye_rect[0], right_eye_rect[1]), 
                                 (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), 
                                 (0, 255, 0), 1)
                
                if left_pupil and left_eye_rect:
                    cv2.circle(display_frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 
                              3, (0, 0, 255), -1)
                
                if right_pupil and right_eye_rect:
                    cv2.circle(display_frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 
                              3, (0, 0, 255), -1)
            
            # Add eye metrics to the display
            cv2.putText(display_frame, f"Blink Rate: {eye_metrics['blink_rate']:.1f} bpm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show gaze direction (use pupil tracking result if available)
            if pupil_tracker and pupil_gaze_direction != "Unknown":
                cv2.putText(display_frame, f"Gaze: {pupil_gaze_direction}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, f"Gaze: {eye_metrics['gaze_direction']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add head posture information
            head_color = (0, 255, 0) if head_direction == "Looking at Screen" else (0, 0, 255)
            cv2.putText(display_frame, f"Head: {head_direction}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_color, 2)
            
            # Add mask status if enabled
            if mask_detector:
                mask_color = (0, 255, 0) if mask_metrics["mask_detected"] else (0, 0, 255)
                cv2.putText(display_frame, f"Mask: {mask_status}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mask_color, 2)
                y_offset = 150
            else:
                y_offset = 120
            
            # Add face count information
            if face_detector:
                face_color = (0, 0, 255) if face_data['multiple_faces_detected'] else (0, 255, 0)
                cv2.putText(display_frame, f"Faces Detected: {face_data['face_count']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
                y_offset += 30
            
            # Add warning for abnormal eye behavior
            if eye_metrics.get("blink_rate_warning", False):
                cv2.putText(display_frame, "ABNORMAL BLINKING!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y_offset += 30
            
            if eye_metrics.get("direction_warning", False):
                cv2.putText(display_frame, f"LOOKING {eye_metrics['gaze_direction']} TOO LONG!", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y_offset += 30
            
            # Show blink count
            cv2.putText(display_frame, f"Blinks: {eye_metrics['blinks']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show EAR if calibrated
            if eye_metrics.get('calibrated', False):
                ear_value = eye_metrics.get('ear', 0)
                ear_color = (0, 0, 255) if ear_value < eye_monitor.EYE_AR_THRESH else (0, 255, 0)
                cv2.putText(display_frame, f"EAR: {ear_value:.4f}", (frame.shape[1] - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        
        # Add FPS counter
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, frame_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Add cheating detection warning
        if cheat_metrics['cheating_detected']:
            cv2.putText(display_frame, "WARNING: Potential cheating detected!", 
                       (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add mask warning if needed
        if mask_detector and mask_metrics["no_mask_detected"]:
            cv2.putText(display_frame, "WARNING: No mask detected!", 
                       (10, frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add multiple faces warning if needed
        if face_detector and face_data['multiple_faces_detected']:
            cv2.putText(display_frame, "WARNING: Multiple faces detected!", 
                       (10, frame_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Log to console only every 1 second
        if time.time() - last_log_time > 1.0:
            last_log_time = time.time()
            
            # Use pupil tracking gaze direction if available
            gaze_info = pupil_gaze_direction if (pupil_tracker and pupil_gaze_direction != "Unknown") else eye_metrics['gaze_direction']
            
            log_str = f"FPS: {fps:.1f} | Blinks: {eye_metrics['blinks']} | Blink rate: {eye_metrics['blink_rate']:.1f} bpm | Gaze: {gaze_info} | Head: {head_direction}"
            
            # Add face count to log
            if face_detector:
                log_str += f" | Faces: {face_data['face_count']}"
            
            # Add mask info if enabled
            if mask_detector:
                log_str += f" | Mask: {mask_status}"
            
            print(log_str)
            
            # Print cheating detection results
            if cheat_metrics['cheating_detected']:
                print(f"Detected potentially prohibited items: {cheat_metrics['cheating_objects']}")
            
            # Print multiple faces warning
            if face_detector and face_data['multiple_faces_detected']:
                print(f"WARNING: Multiple faces detected ({face_data['face_count']})")
        
        # Display the frame
        cv2.imshow("Monitoring System", display_frame)
        
        # Save to video file if specified
        if video_writer:
            video_writer.write(display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Toggle display mode - include mask mode if enabled
            if mask_detector:
                modes = ['split', 'eye', 'cheat', 'head', 'pupil', 'mask', 'faces', 'combined'] if pupil_tracker else ['split', 'eye', 'cheat', 'head', 'pupil', 'mask', 'faces', 'combined']
            else:
                modes = ['split', 'eye', 'cheat', 'head', 'pupil', 'faces', 'combined'] if pupil_tracker else ['split', 'eye', 'cheat', 'head', 'faces', 'combined']
            current_idx = modes.index(display_mode) if display_mode in modes else 0
            display_mode = modes[(current_idx + 1) % len(modes)]
            print(f"Display mode changed to: {display_mode}")
        elif key == ord('c'):
            # Force recalibration for both eye and head
            eye_monitor.calibrated = False
            eye_monitor.baseline_ear = None
            eye_monitor.baseline_samples = []
            
            # Reset head calibration
            head_calibration_phase = True
            head_calibration_start = time.time()
            head_calibration_frames = []
            calibrated_head_angles = None
            
            print("Forced recalibration. Please look at the camera.")
    
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
    print(f"- Calibrated EAR threshold: {stats['threshold']:.3f}")
    print(f"- Baseline EAR: {stats['baseline_ear']:.3f}")
    
    # Get cheating detection summary
    cheat_stats = cheating_detector.get_statistics()
    print(f"- Cheating instances detected: {cheat_stats['total_cheating_instances']}")
    print(f"- Total cheating time: {cheat_stats['total_cheating_time']:.2f} seconds")
    print(f"- Cheating percentage: {cheat_stats['cheating_percentage']:.2f}%")
    
    # Show mask detection statistics if enabled
    if mask_detector:
        mask_stats = mask_detector.get_statistics()
        print(f"- Total frames processed for mask detection: {mask_stats['total_frames']}")
        print(f"- Frames without mask: {mask_stats['frames_without_mask']}")
        print(f"- Mask compliance rate: {mask_stats['mask_compliance_rate']:.2f}%")
    
    # Show pupil tracking statistics if enabled
    if pupil_tracker:
        print(f"- Pupil tracking summary:")
        print(f"  - Calibration status: {'Completed' if pupil_tracker.calibration_complete else 'Not completed'}")
        if pupil_tracker.calibration_complete:
            print(f"  - Number of calibration points: {len(pupil_tracker.center_points)}")
    
    # Show face detection statistics if enabled
    if face_detector:
        face_stats = face_detector.get_statistics()
        print(f"- Face detection summary:")
        print(f"  - Total frames processed: {face_stats['total_frames']}")
        print(f"  - Total faces detected: {face_stats['total_faces_detected']}")
        print(f"  - Average faces per frame: {face_stats['average_faces_per_frame']:.2f}")
        print(f"  - Maximum faces detected: {face_stats['max_faces_detected']}")
        print(f"  - Face count distribution:")
        for count, frequency in face_stats['face_count_distribution'].items():
            print(f"    - {count} faces: {frequency} frames")

if __name__ == "__main__":
    main()