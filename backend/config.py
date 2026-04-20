"""Configuration and constants for the Korean OCR application."""

import os
import json

# OCR Configuration
OCR_LANGUAGE = "korean"

# Gemini API Configuration
# Set your API key via environment variable: GEMINI_API_KEY
GEMINI_MODEL = "gemini-2.5-flash"

# GUI Configuration
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BRUSH_SIZE = 3
BRUSH_COLOR = (0, 255, 0)  # Green

# Image Processing
MAX_IMAGE_SIZE = 2600  # Max dimension for image
RECTANGLE_PADDING = 0  # Padding around detected region in pixels

# Detection & Recognition Confidence Thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.5 # Filter detection bboxes with dt_scores > this threshold
RECOGNITION_CONFIDENCE_THRESHOLD = 0.6  # Filter recognized text with rec_score > this threshold

# Text Replacement Configuration
CUSTOM_FONT_PATH = "/mnt/data/manhwa-machine-translator/CCAskForMercy-Regular.ttf"
TEXT_FONT_SIZE = 22  # Font size in pixels
TEXT_FONT_COLOR = (0, 0, 0)  # RGB: Black text
TEXT_STROKE_WIDTH = 3  # Thick outline so text cuts through arbitrary backgrounds
TEXT_STROKE_COLOR = (255, 255, 255)  # RGB: White outline
TEXT_LINE_HEIGHT_MULTIPLIER = 1.0  # Line height multiplier

# Last opened path storage
LAST_OPENED_PATH_FILE = ".last_opened_path.json"


def get_last_opened_path():
    """Get the last opened file or folder path."""
    try:
        if os.path.exists(LAST_OPENED_PATH_FILE):
            with open(LAST_OPENED_PATH_FILE, "r") as f:
                data = json.load(f)
                return data.get("path", os.path.expanduser("~"))
    except Exception as e:
        print(f"Failed to read last opened path: {e}")
    return os.path.expanduser("~")


def save_last_opened_path(path):
    """Save the last opened file or folder path."""
    try:
        with open(LAST_OPENED_PATH_FILE, "w") as f:
            json.dump({"path": path}, f)
    except Exception as e:
        print(f"Failed to save last opened path: {e}")
