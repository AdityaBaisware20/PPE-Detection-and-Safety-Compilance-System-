import cv2
import threading
import time
import pygame
from datetime import datetime
import os

class PPEAlertSystem:
    def __init__(self, enable_sound=True, alert_cooldown=5, sound_file_path=None):
        """
        Initialize the PPE Alert System
        
        Args:
            enable_sound (bool): Enable sound alerts
            alert_cooldown (int): Cooldown period between alerts in seconds
            sound_file_path (str): Path to MP3 alert sound file
        """
        self.enable_sound = enable_sound
        self.alert_cooldown = alert_cooldown
        self.sound_file_path = sound_file_path
        self.last_alert_time = {}
        self.alert_active = False
        
        # Initialize pygame mixer for sound alerts
        if self.enable_sound:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.load_alert_sound()
            except Exception as e:
                print(f"Warning: Could not initialize sound system: {e}")
                self.enable_sound = False
    
    def load_alert_sound(self):
        """Load alert sound file"""
        if self.sound_file_path and os.path.exists(self.sound_file_path):
            try:
                # Load MP3 file
                self.alert_sound = pygame.mixer.Sound(self.sound_file_path)
                print(f"Successfully loaded alert sound: {self.sound_file_path}")
            except Exception as e:
                print(f"Error loading sound file {self.sound_file_path}: {e}")
                # Fallback to generated beep sound
                self.generate_beep_sound()
        else:
            if self.sound_file_path:
                print(f"Warning: Sound file not found: {self.sound_file_path}")
            print("Using generated beep sound as fallback")
            # Fallback to generated beep sound
            self.generate_beep_sound()
    
    def generate_beep_sound(self):
        """Generate a simple beep sound as fallback"""
        try:
            # Create a simple beep sound
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            
            frames = int(duration * sample_rate)
            arr = []
            for i in range(frames):
                wave = 4096 * (i % (sample_rate // frequency) < (sample_rate // frequency) // 2) - 2048
                arr.append([wave, wave])
            
            sound_array = pygame.sndarray.make_sound(pygame.array.array('i', arr))
            self.alert_sound = sound_array
            print("Generated fallback beep sound")
        except Exception as e:
            print(f"Warning: Could not generate alert sound: {e}")
            self.enable_sound = False
    
    def check_ppe_violations(self, results, class_names):
        """
        Check for PPE violations in the detection results
        
        Args:
            results: YOLO detection results
            class_names: List of class names
            
        Returns:
            dict: Dictionary containing violation information
        """
        violations = {
            'persons_without_hardhat': [],
            'persons_without_mask': [],
            'persons_without_safety_vest': [],
            'total_violations': 0
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                cls_id = int(box.cls)
                class_name = class_names[cls_id]
                conf = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check for PPE violations
                if class_name == 'NO-Hardhat' and conf > 0.5:
                    violations['persons_without_hardhat'].append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
                    violations['total_violations'] += 1
                
                elif class_name == 'NO-Mask' and conf > 0.5:
                    violations['persons_without_mask'].append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
                    violations['total_violations'] += 1
                
                elif class_name == 'NO-Safety Vest' and conf > 0.5:
                    violations['persons_without_safety_vest'].append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
                    violations['total_violations'] += 1
        
        return violations
    
    def should_trigger_alert(self, violation_type):
        """
        Check if enough time has passed since the last alert for this violation type
        
        Args:
            violation_type (str): Type of violation
            
        Returns:
            bool: True if alert should be triggered
        """
        current_time = time.time()
        if violation_type not in self.last_alert_time:
            self.last_alert_time[violation_type] = current_time
            return True
        
        if current_time - self.last_alert_time[violation_type] >= self.alert_cooldown:
            self.last_alert_time[violation_type] = current_time
            return True
        
        return False
    
    def play_sound_alert(self):
        """Play sound alert"""
        if self.enable_sound and hasattr(self, 'alert_sound'):
            try:
                # Stop any currently playing sound to avoid overlapping
                pygame.mixer.stop()
                self.alert_sound.play()
                print("🔊 Playing alert sound")
            except Exception as e:
                print(f"Error playing sound alert: {e}")
    
    def play_sound_async(self):
        """Play sound alert in a separate thread to avoid blocking"""
        if self.enable_sound and hasattr(self, 'alert_sound'):
            def play_sound():
                try:
                    # Stop any currently playing sound
                    pygame.mixer.stop()
                    self.alert_sound.play()
                    print("🔊 Playing alert sound (async)")
                except Exception as e:
                    print(f"Error playing sound alert: {e}")
            
            # Play sound in separate thread
            sound_thread = threading.Thread(target=play_sound)
            sound_thread.daemon = True
            sound_thread.start()
    
    def show_visual_alert(self, frame, violations):
        """
        Show visual alert on the frame
        
        Args:
            frame: OpenCV frame
            violations: Dictionary containing violation information
            
        Returns:
            frame: Modified frame with alert overlay
        """
        if violations['total_violations'] > 0:
            # Create alert overlay
            overlay = frame.copy()
            
            # Red warning background
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Alert text
            alert_text = f"PPE VIOLATION DETECTED! ({violations['total_violations']} violations)"
            cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Detailed violation info
            y_offset = 55
            if violations['persons_without_hardhat']:
                cv2.putText(frame, f"Missing Hardhat: {len(violations['persons_without_hardhat'])}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            if violations['persons_without_mask']:
                cv2.putText(frame, f"Missing Mask: {len(violations['persons_without_mask'])}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            if violations['persons_without_safety_vest']:
                cv2.putText(frame, f"Missing Safety Vest: {len(violations['persons_without_safety_vest'])}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Highlight violation areas with pulsing effect
            pulse = int(abs(time.time() * 5) % 2)
            if pulse:
                for violation_list in [violations['persons_without_hardhat'], 
                                     violations['persons_without_mask'], 
                                     violations['persons_without_safety_vest']]:
                    for violation in violation_list:
                        x1, y1, x2, y2 = violation['bbox']
                        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 3)
        
        return frame
    
    def log_violation(self, violations):
        """
        Log violations to a file
        
        Args:
            violations: Dictionary containing violation information
        """
        if violations['total_violations'] > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] PPE Violations Detected:\n"
            
            if violations['persons_without_hardhat']:
                log_entry += f"  - Missing Hardhat: {len(violations['persons_without_hardhat'])} persons\n"
            if violations['persons_without_mask']:
                log_entry += f"  - Missing Mask: {len(violations['persons_without_mask'])} persons\n"
            if violations['persons_without_safety_vest']:
                log_entry += f"  - Missing Safety Vest: {len(violations['persons_without_safety_vest'])} persons\n"
            
            log_entry += f"  - Total Violations: {violations['total_violations']}\n\n"
            
            # Write to log file
            try:
                with open('ppe_violations.log', 'a') as log_file:
                    log_file.write(log_entry)
            except Exception as e:
                print(f"Error writing to log file: {e}")
    
    def trigger_alert(self, frame, violations):
        """
        Main method to trigger alerts based on violations
        
        Args:
            frame: OpenCV frame
            violations: Dictionary containing violation information
            
        Returns:
            frame: Modified frame with alert overlay
        """
        if violations['total_violations'] > 0:
            # Check if we should trigger alerts (considering cooldown)
            should_alert = False
            
            if violations['persons_without_hardhat'] and self.should_trigger_alert('hardhat'):
                should_alert = True
            if violations['persons_without_mask'] and self.should_trigger_alert('mask'):
                should_alert = True
            if violations['persons_without_safety_vest'] and self.should_trigger_alert('safety_vest'):
                should_alert = True
            
            if should_alert:
                # Play sound alert (async to avoid blocking)
                self.play_sound_async()
                
                # Log violation
                self.log_violation(violations)
                
                # Print alert to console
                print(f"⚠️  PPE VIOLATION ALERT! - {violations['total_violations']} violations detected")
            
            # Always show visual alert (regardless of cooldown)
            frame = self.show_visual_alert(frame, violations)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.enable_sound:
            try:
                pygame.mixer.quit()
            except:
                pass