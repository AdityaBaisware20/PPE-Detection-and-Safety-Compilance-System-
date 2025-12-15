import argparse

import cv2
from ultralytics import YOLO
import numpy as np
import time
import threading
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import os
import json
import pandas as pd
from config import *

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

class PPEAlertSystem:
    def __init__(self, enable_sound=ENABLE_SOUND, alert_cooldown=ALERT_COOLDOWN, sound_file_path=SOUND_FILE_PATH):
        """
        Initialize the PPE Alert System
        
        Args:
            enable_sound (bool): Enable sound alerts (now handled client-side)
            alert_cooldown (int): Cooldown period between alerts in seconds
            sound_file_path (str): Path to MP3 alert sound file
        """
        self.enable_sound = enable_sound
        self.alert_cooldown = alert_cooldown
        self.sound_file_path = sound_file_path
        self.last_alert_time = {}
        self.alert_active = False
        self.alert_callback = None
        self.violation_stats = {
            'hardhat': 0,
            'mask': 0,
            'safety_vest': 0,
            'total': 0
        }
        self.seen_violations = {
            'hardhat': set(),
            'mask': set(),
            'safety_vest': set()
        }
        self.active_violations = {
            'hardhat': set(),
            'mask': set(),
            'safety_vest': set()
        }
        self.violation_log = []  # Store violation events with timestamps for CSV export

    def check_ppe_violations(self, results, class_names):
        """
        Check for PPE violations in the detection results
        
        Args:
            results: YOLO detection results
            class_names: List of class names
            
        Returns:
            dict: Dictionary containing violation information
        """
        # Reset active violations for the current frame
        self.active_violations = {
            'hardhat': set(),
            'mask': set(),
            'safety_vest': set()
        }
        
        violations = {
            'persons_without_hardhat': [],
            'persons_without_mask': [],
            'persons_without_safety_vest': [],
            'total_violations': 0
        }
        
        # Process current frame detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                cls_id = int(box.cls)
                class_name = class_names[cls_id]
                conf = float(box.conf)
                bbox = [int(x) for x in box.xyxy[0]]
                track_id = int(box.id) if box.id is not None else None
                
                if class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and conf > CONFIDENCE_THRESHOLD:
                    violation_map = {
                        'NO-Hardhat': 'hardhat',
                        'NO-Mask': 'mask',
                        'NO-Safety Vest': 'safety_vest'
                    }
                    violation_type = violation_map[class_name]
                    
                    if track_id is not None:
                        self.active_violations[violation_type].add(track_id)
                        
                        if track_id not in self.seen_violations[violation_type]:
                            self.seen_violations[violation_type].add(track_id)
                            self.violation_stats[violation_type] += 1
                            self.violation_stats['total'] += 1
                            new_log = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'violation_type': violation_type,
                                'count': 1,
                                'person_id': track_id
                            }
                            self.violation_log.append(new_log)
                            
                            if self.alert_callback:
                                message = json.dumps({
                                    "type": "violation",
                                    "data": new_log
                                })
                                self.alert_callback(message)
                    
                    violations[f'persons_without_{violation_type}'].append({
                        'bbox': bbox,
                        'confidence': conf,
                        'id': track_id
                    })
        
        # Calculate total current violations
        for v_type in ['hardhat', 'mask', 'safety_vest']:
            violations['total_violations'] += len(self.active_violations[v_type])
        
        if self.alert_callback and violations['total_violations'] > 0:
            stats_message = json.dumps({
                "type": "stats",
                "stats": self.violation_stats
            })
            self.alert_callback(stats_message)
        
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

    def show_visual_alert(self, frame, violations):
        """
        Show a professional and elegant alert card in the top-left of the frame
        
        Args:
            frame: OpenCV frame
            violations: Dictionary containing violation information
            
        Returns:
            frame: Modified frame with alert card
        """
        if violations['total_violations'] > 0:
            card_height = ALERT_CARD_LINE_HEIGHT * ALERT_CARD_NUM_LINES + 20

            overlay = frame.copy()
            cv2.rectangle(overlay, (ALERT_CARD_START_X + 3, ALERT_CARD_START_Y + 3), 
                          (ALERT_CARD_START_X + ALERT_CARD_WIDTH + 3, ALERT_CARD_START_Y + card_height + 3), 
                          ALERT_CARD_SHADOW_COLOR, -1)
            cv2.rectangle(overlay, (ALERT_CARD_START_X, ALERT_CARD_START_Y), 
                          (ALERT_CARD_START_X + ALERT_CARD_WIDTH, ALERT_CARD_START_Y + card_height), 
                          ALERT_CARD_BG_COLOR, -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

            cv2.rectangle(frame, (ALERT_CARD_START_X, ALERT_CARD_START_Y), 
                          (ALERT_CARD_START_X + ALERT_CARD_WIDTH, ALERT_CARD_START_Y + card_height), 
                          ALERT_CARD_BORDER_COLOR, 1, lineType=cv2.LINE_AA)

            icon_x, icon_y = ALERT_CARD_START_X + 10, ALERT_CARD_START_Y + 15
            cv2.polylines(frame, [np.array([[icon_x, icon_y], 
                                            [icon_x + 10, icon_y], 
                                            [icon_x + 5, icon_y - 10]])], 
                          True, ALERT_CARD_ACCENT_COLOR, 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "!", (icon_x + 4, icon_y - 2), getattr(cv2, ALERT_CARD_FONT), 
                        0.4, ALERT_CARD_ACCENT_COLOR, 1, lineType=cv2.LINE_AA)

            header_text = "PPE Violations"
            cv2.putText(frame, header_text, (ALERT_CARD_START_X + 25, ALERT_CARD_START_Y + 15), 
                        getattr(cv2, ALERT_CARD_FONT), ALERT_CARD_FONT_SCALE, ALERT_CARD_TEXT_COLOR, ALERT_CARD_THICKNESS, lineType=cv2.LINE_AA)

            rows = [
                (f"No Hardhat: {len(violations['persons_without_hardhat'])}"),
                (f"No Mask: {len(violations['persons_without_mask'])}"),
                (f"No Vest: {len(violations['persons_without_safety_vest'])}"),
                (f"Total: {violations['total_violations']}")
            ]

            for i, text in enumerate(rows):
                y = ALERT_CARD_START_Y + 35 + i * ALERT_CARD_LINE_HEIGHT
                cv2.putText(frame, text, (ALERT_CARD_START_X + 10, y), 
                            getattr(cv2, ALERT_CARD_FONT), ALERT_CARD_FONT_SCALE, ALERT_CARD_TEXT_COLOR, ALERT_CARD_THICKNESS, lineType=cv2.LINE_AA)
        
        return frame

    def log_violation(self, violations):
        """
        Log violations to a file
        
        Args:
            violations: Dictionary containing violation information
        
        Returns:
            str: The log entry
        """
        if violations['total_violations'] > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] PPE Violations Detected:\n"
            
            if violations['persons_without_hardhat']:
                log_entry += f" - Missing Hardhat: {len(violations['persons_without_hardhat'])} persons\n"
            if violations['persons_without_mask']:
                log_entry += f" - Missing Mask: {len(violations['persons_without_mask'])} persons\n"
            if violations['persons_without_safety_vest']:
                log_entry += f" - Missing Safety Vest: {len(violations['persons_without_safety_vest'])} persons\n"
            
            log_entry += f" - Total Violations: {violations['total_violations']}\n\n"
            
            try:
                with open('ppe_violations.log', 'a') as log_file:
                    log_file.write(log_entry)
            except Exception as e:
                print(f"Error writing to log file: {e}")
            
            return log_entry
        
        return ""

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
            frame = self.show_visual_alert(frame, violations)
            should_alert = False
            
            if violations['persons_without_hardhat'] and self.should_trigger_alert('hardhat'):
                should_alert = True
            if violations['persons_without_mask'] and self.should_trigger_alert('mask'):
                should_alert = True
            if violations['persons_without_safety_vest'] and self.should_trigger_alert('safety_vest'):
                should_alert = True
            
            if should_alert:
                log_entry = self.log_violation(violations)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_path = f"/screenshots/{timestamp}.jpg"
                cv2.imwrite(screenshot_path[1:], frame)
                
                if self.alert_callback:
                    message = json.dumps({
                        "type": "alert",
                        "log": log_entry,
                        "image": screenshot_path
                    })
                    self.alert_callback(message)
                
                print(f"⚠️ PPE VIOLATION ALERT! - {violations['total_violations']} violations detected")
        
        return frame

    def cleanup(self):
        """Clean up resources"""
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description="PPE Detection using YOLO with Alert System on FastAPI")
    parser.add_argument('--source', type=str, default='webcam',
                        help='Input source: RTSP URL, video file, or "webcam" for default camera (default: webcam)')
    parser.add_argument('--webcam-id', type=int, default=0,
                        help='Webcam device ID (default: 0 for primary camera)')
    parser.add_argument('--classes', nargs='+', default=CLASS_NAMES,
                        help='List of classes to detect (e.g., Hardhat Mask)')
    parser.add_argument('--enable-sound', action='store_true', default=ENABLE_SOUND, help='Enable sound alerts on client')
    parser.add_argument('--sound-file', type=str, default=SOUND_FILE_PATH,
                        help='Path to MP3 alert sound file (e.g., alert.mp3)')
    parser.add_argument('--alert-cooldown', type=int, default=ALERT_COOLDOWN, help='Cooldown period between alerts in seconds')
    return parser.parse_args()

def draw_bounding_boxes(frame, results, class_names, filter_classes, violations):
    """
    Draw bounding boxes for detected objects and violations
    
    Args:
        frame: OpenCV frame
        results: YOLO detection results
        class_names: List of class names
        filter_classes: List of classes to display
        violations: Dictionary containing violation information
    
    Returns:
        frame: Frame with bounding boxes drawn
    """
    # Draw violation boxes
    for violation_type in ['hardhat', 'mask', 'safety_vest']:
        for violation in violations[f'persons_without_{violation_type}']:
            x1, y1, x2, y2 = violation['bbox']
            label = f"No {violation_type.replace('safety_vest', 'Vest')} ID:{violation.get('id', 'N/A')}"
            color = (0, 0, 255)  # Red for violations
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw boxes for non-violation classes
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        
        for box in boxes:
            cls_id = int(box.cls)
            class_name = class_names[cls_id]
            if class_name not in filter_classes or class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                continue  # Skip violation classes as they're handled above
            conf = float(box.conf)
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            track_id = int(box.id) if box.id is not None else 'N/A'
            label = f"{class_name} {conf:.2f} ID:{track_id}"
            
            if class_name in ['Hardhat', 'Mask', 'Safety Vest']:
                color = (0, 255, 0)  # Green for safety equipment
            else:
                color = (255, 0, 0)  # Blue for other objects (Person, Safety Cone, etc.)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

app = FastAPI()
manager = ConnectionManager()
latest_frame = None
loop = None
capture_thread = None
args = parse_arguments()
os.makedirs('screenshots', exist_ok=True)
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")
model = YOLO('best.pt')
alert_system = PPEAlertSystem(
    enable_sound=args.enable_sound,
    alert_cooldown=args.alert_cooldown,
    sound_file_path=args.sound_file
)
filter_classes = [cls for cls in args.classes if cls in CLASS_NAMES]
if not filter_classes:
    print("Warning: No valid classes provided. Using all classes.")
    filter_classes = CLASS_NAMES
print(f"Detecting classes: {filter_classes}")
print(f"Alert system enabled with {args.alert_cooldown}s cooldown")
if args.enable_sound:
    if args.sound_file:
        print(f"Using custom alert sound on client: {args.sound_file}")
    else:
        print("Using generated beep sound on client")
if args.source.lower() == 'webcam':
    source = args.webcam_id
    print(f"Using webcam device ID: {args.webcam_id}")
else:
    source = args.source
    print(f"Using source: {args.source}")
def capture_loop():
    global latest_frame
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
            violations = alert_system.check_ppe_violations(results, CLASS_NAMES)
            annotated_frame = draw_bounding_boxes(frame, results, CLASS_NAMES, filter_classes, violations)
            annotated_frame = alert_system.trigger_alert(annotated_frame, violations)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                latest_frame = buffer.tobytes()
            
            time.sleep(0.03)
    finally:
        cap.release()
        alert_system.cleanup()
@app.on_event("startup")
def startup():
    global loop
    loop = asyncio.get_event_loop()
    def broadcast_alert(message):
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), loop)
    alert_system.alert_callback = broadcast_alert
@app.get("/video")
async def video():
    async def gen_frames():
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            await asyncio.sleep(0.03)
    
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "start":
                global capture_thread
                if capture_thread is None:
                    capture_thread = threading.Thread(target=capture_loop, daemon=True)
                    capture_thread.start()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
@app.get("/export/csv")
async def export_csv():
    data = alert_system.violation_log
    df = pd.DataFrame(data)
    csv_content = df.to_csv(index=False)
    return Response(content=csv_content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=violation_report.csv"})
@app.get("/")
async def root():
    beep_script = ""
    audio_tag = ""
    play_command = ""
    if args.enable_sound:
        if args.sound_file:
            audio_tag = '<audio id="alert_sound" src="/alert_sound" preload="auto"></audio>'
            play_command = 'document.getElementById("alert_sound").play().catch(e => console.error("Error playing sound:", e));'
        else:
            beep_script = """
            function playBeep() {
                var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                var oscillator = audioCtx.createOscillator();
                oscillator.type = 'sine';
                oscillator.frequency.setValueAtTime(800, audioCtx.currentTime);
                oscillator.connect(audioCtx.destination);
                oscillator.start();
                setTimeout(function() {
                    oscillator.stop();
                }, 500);
            }
            """
            play_command = 'playBeep();'
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPE Detection Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/moment@2.29.4/moment.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f2f5;
            }}
            .header {{
                background-color: #007bff;
                color: white;
                padding: 15px;
                text-align: center;
                font-size: 24px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .dashboard {{
                display: flex;
                height: calc(100vh - 60px);
            }}
            .main-panel {{
                flex: 1;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .side-panel {{
                width: 400px;
                background-color: #ffffff;
                padding: 20px;
                overflow-y: auto;
                border-left: 1px solid #dee2e6;
                box-shadow: -2px 0 5px rgba(0,0,0,0.05);
            }}
            h1, h2 {{
                color: #333;
            }}
            h2 {{
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            button {{
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                cursor: pointer;
                font-size: 16px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            button:hover {{
                background-color: #218838;
            }}
            button:disabled {{
                background-color: #6c757d;
                cursor: not-allowed;
            }}
            .alert-item {{
                background-color: #fff;
                padding: 15px;
                margin-bottom: 15px;
                border: 1px solid #dee2e6;
                border-left: 5px solid #dc3545;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                white-space: pre-wrap;
                font-size: 14px;
                transition: transform 0.2s;
            }}
            .alert-item:hover {{
                transform: translateY(-2px);
            }}
            img.video-feed {{
                width: 100%;
                max-width: 800px;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .alert-item img {{
                width: 100%;
                margin-top: 10px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .stats-card {{
                background-color: #fff;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .stats-card h3 {{
                margin: 0 0 10px 0;
                font-size: 18px;
                color: #333;
                border-bottom: 1px solid #007bff;
                padding-bottom: 5px;
            }}
            .stats-card p {{
                margin: 5px 0;
                font-size: 14px;
                color: #333;
            }}
            .stats-card .count {{
                font-weight: bold;
                color: #dc3545;
            }}
            #download-btn {{
                background-color: #007bff;
                margin-top: 10px;
            }}
            #download-btn:hover {{
                background-color: #0056b3;
            }}
            #violation-chart {{
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">PPE Detection System</div>
        <div class="dashboard">
            <div class="main-panel">
                <h1>Live Feed</h1>
                <button id="start-btn">Start Detection</button>
                <img id="video-feed" class="video-feed" src="/video" alt="Live Feed">
            </div>
            <div class="side-panel">
                <h2>Violation Statistics</h2>
                <div class="stats-card" id="stats-card">
                    <h3>Unique Violations</h3>
                    <p>No Hardhat: <span class="count" id="hardhat-count">0</span></p>
                    <p>No Mask: <span class="count" id="mask-count">0</span></p>
                    <p>No Safety Vest: <span class="count" id="vest-count">0</span></p>
                    <p>Total: <span class="count" id="total-count">0</span></p>
                    <button id="download-btn">Download CSV Report</button>
                </div>
                <canvas id="violation-chart" width="400" height="200"></canvas>
                <h2>Violation Trends</h2>
                <canvas id="timeline-chart" width="400" height="200"></canvas>
                <h2>Alert Log</h2>
                <div id="alert-list"></div>
            </div>
        </div>
        {audio_tag}
        <script>
            {beep_script}
            const ws = new WebSocket("ws://" + location.host + "/ws");
            document.getElementById("start-btn").onclick = function() {{
                ws.send("start");
                this.disabled = true;
            }};
            let violationChart;
            const chartData = {{
                labels: ['No Hardhat', 'No Mask', 'No Safety Vest'],
                datasets: [{{
                    label: 'Unique Violations',
                    data: [0, 0, 0],
                    backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)'],
                    borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)'],
                    borderWidth: 1
                }}]
            }};
            const chartConfig = {{
                type: 'bar',
                data: chartData,
                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }};
            violationChart = new Chart(document.getElementById('violation-chart'), chartConfig);
            
            let timelineChart;
            const timelineData = {{
                labels: [],
                datasets: [{{
                    label: 'Hardhat Violations',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    fill: false
                }}, {{
                    label: 'Mask Violations',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    fill: false
                }}, {{
                    label: 'Safety Vest Violations',
                    data: [],
                    borderColor: 'rgba(255, 206, 86, 1)',
                    fill: false
                }}]
            }};
            const timelineConfig = {{
                type: 'line',
                data: timelineData,
                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }};
            timelineChart = new Chart(document.getElementById('timeline-chart'), timelineConfig);
            
            let logs = [];
            
            function updateTimeline() {{
                const buckets = {{}};
                logs.forEach(log => {{
                    const hour = moment(log.timestamp, "YYYY-MM-DD HH:mm:ss").format('YYYY-MM-DD HH:00');
                    if (!buckets[hour]) {{
                        buckets[hour] = {{hardhat: 0, mask: 0, safety_vest: 0}};
                    }}
                    buckets[hour][log.violation_type] += log.count;
                }});
                const sortedKeys = Object.keys(buckets).sort();
                timelineChart.data.labels = sortedKeys;
                timelineChart.data.datasets[0].data = sortedKeys.map(k => buckets[k].hardhat || 0);
                timelineChart.data.datasets[1].data = sortedKeys.map(k => buckets[k].mask || 0);
                timelineChart.data.datasets[2].data = sortedKeys.map(k => buckets[k].safety_vest || 0);
                timelineChart.update();
            }}
            
            ws.onmessage = function(event) {{
                const message = JSON.parse(event.data);
                if (message.type === "stats") {{
                    const data = message.stats;
                    document.getElementById("hardhat-count").textContent = data.hardhat;
                    document.getElementById("mask-count").textContent = data.mask;
                    document.getElementById("vest-count").textContent = data.safety_vest;
                    document.getElementById("total-count").textContent = data.total;
                    violationChart.data.datasets[0].data = [data.hardhat, data.mask, data.safety_vest];
                    violationChart.update();
                }} else if (message.type === "alert") {{
                    var alertList = document.getElementById("alert-list");
                    var alertDiv = document.createElement("div");
                    alertDiv.className = "alert-item";
                    alertDiv.textContent = message.log;
                    if (message.image) {{
                        var img = document.createElement("img");
                        img.src = message.image;
                        alertDiv.appendChild(img);
                    }}
                    alertList.appendChild(alertDiv);
                    alertList.scrollTop = alertList.scrollHeight;
                    {play_command}
                }} else if (message.type === "violation") {{
                    logs.push(message.data);
                    updateTimeline();
                }}
            }};
            document.getElementById("download-btn").onclick = function() {{
                window.location.href = "/export/csv";
            }};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
if args.sound_file:
    @app.get("/alert_sound")
    def get_alert_sound():
        return FileResponse(args.sound_file, media_type="audio/mpeg")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)