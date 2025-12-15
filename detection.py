# updated with alert integration, webcam support, and MP3 sound file support
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from alert_system import PPEAlertSystem

def parse_arguments():
    parser = argparse.ArgumentParser(description="PPE Detection using YOLO with Alert System")
    parser.add_argument('--source', type=str, default='webcam', 
                       help='Input source: RTSP URL, video file, image file, or "webcam" for default camera (default: webcam)')
    parser.add_argument('--webcam-id', type=int, default=0, 
                       help='Webcam device ID (default: 0 for primary camera)')
    parser.add_argument('--classes', nargs='+', default=['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                                                        'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'],
                        help='List of classes to detect (e.g., Hardhat Mask)')
    parser.add_argument('--enable-sound', action='store_true', help='Enable sound alerts')
    parser.add_argument('--sound-file', type=str, default=None,
                       help='Path to MP3 alert sound file (e.g., alert.mp3)')
    parser.add_argument('--alert-cooldown', type=int, default=5, help='Cooldown period between alerts in seconds')
    return parser.parse_args()

def draw_bounding_boxes(frame, results, class_names, filter_classes):
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
           
        for box in boxes:
            cls_id = int(box.cls)
            class_name = class_names[cls_id]
            if class_name not in filter_classes:
                continue
            conf = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_name} {conf:.2f}"
           
            # Color coding: Green for safety equipment, Red for violations, Blue for others
            if class_name in ['Hardhat', 'Mask', 'Safety Vest']:
                color = (0, 255, 0) # Green for safety equipment
            elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                color = (0, 0, 255) # Red for violations
            else:
                color = (255, 0, 0) # Blue for other objects
           
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
   
    return frame

def process_image(model, image_path, filter_classes, class_names, alert_system):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
   
    results = model(frame)
   
    # Check for PPE violations
    violations = alert_system.check_ppe_violations(results, class_names)
   
    # Draw bounding boxes
    annotated_frame = draw_bounding_boxes(frame, results, class_names, filter_classes)
   
    # Trigger alerts if violations found
    annotated_frame = alert_system.trigger_alert(annotated_frame, violations)
   
    # Save and display result
    cv2.imwrite('output.jpg', annotated_frame)
    cv2.imshow('PPE Detection with Alerts', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video_or_rtsp(model, source, filter_classes, class_names, alert_system):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return
   
    print("Press 'q' to quit")
   
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
           
            results = model(frame)
           
            # Check for PPE violations
            violations = alert_system.check_ppe_violations(results, class_names)
           
            # Draw bounding boxes
            annotated_frame = draw_bounding_boxes(frame, results, class_names, filter_classes)
           
            # Trigger alerts if violations found
            annotated_frame = alert_system.trigger_alert(annotated_frame, violations)
           
            cv2.imshow('PPE Detection with Alerts', annotated_frame)
           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Clean up alert system resources
        alert_system.cleanup()

def main():
    args = parse_arguments()
   
    # Initialize YOLO model
    model = YOLO('best.pt')
   
    # Initialize alert system with MP3 file path
    alert_system = PPEAlertSystem(
        enable_sound=args.enable_sound,
        alert_cooldown=args.alert_cooldown,
        sound_file_path=args.sound_file
    )
   
    class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                   'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
   
    filter_classes = [cls for cls in args.classes if cls in class_names]
    if not filter_classes:
        print("Warning: No valid classes provided. Using all classes.")
        filter_classes = class_names
   
    print(f"Detecting classes: {filter_classes}")
    print(f"Alert system enabled with {args.alert_cooldown}s cooldown")
    if args.enable_sound:
        if args.sound_file:
            print(f"Using custom alert sound: {args.sound_file}")
        else:
            print("Using generated beep sound")
    
    # Determine the input source
    if args.source.lower() == 'webcam':
        source = args.webcam_id
        print(f"Using webcam device ID: {args.webcam_id}")
    elif args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        process_image(model, args.source, filter_classes, class_names, alert_system)
        return
    else:
        source = args.source
        print(f"Using source: {args.source}")
   
    # Process video/webcam/RTSP stream
    process_video_or_rtsp(model, source, filter_classes, class_names, alert_system)

if __name__ == "__main__":
    main()