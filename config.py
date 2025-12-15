# python .\detection.py --webcam-id 0 --enable-sound --classes Person Hardhat Mask
# Configuration settings for PPE Detection System

# Detection thresholds
IOU_THRESHOLD = 0.9  # IoU threshold for matching bounding boxes
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for violation detection
PERSON_TRACKING_TIMEOUT = 1000  # Seconds before a tracked person is removed

# Alert settings
ALERT_COOLDOWN = 5  # Seconds between consecutive alerts for the same violation type
ENABLE_SOUND = True  # Enable sound alerts on client
SOUND_FILE_PATH = None  # Path to MP3 alert sound file (None for default beep)

# Alert card visual settings
ALERT_CARD_WIDTH = 220
ALERT_CARD_LINE_HEIGHT = 20
ALERT_CARD_START_X = 10
ALERT_CARD_START_Y = 10
ALERT_CARD_NUM_LINES = 5
ALERT_CARD_BG_COLOR = (245, 245, 245, 200)  # Background color (RGBA)
ALERT_CARD_BORDER_COLOR = (200, 0, 0, 255)  # Border color (RGBA)
ALERT_CARD_TEXT_COLOR = (30, 30, 30)  # Text color (RGB)
ALERT_CARD_ACCENT_COLOR = (200, 0, 0)  # Accent color for icon (RGB)
ALERT_CARD_FONT = 'FONT_HERSHEY_SIMPLEX'  # OpenCV font
ALERT_CARD_FONT_SCALE = 0.45
ALERT_CARD_THICKNESS = 1
ALERT_CARD_SHADOW_COLOR = (50, 50, 50, 100)  # Shadow color for overlay (RGBA)

# Class names for detection
CLASS_NAMES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]