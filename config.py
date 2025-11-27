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
RECOGNITION_CONFIDENCE_THRESHOLD = 0.8  # Filter recognized text with rec_score > this threshold

# Text Replacement Configuration
TEXT_FONT_SCALE = 0.5  # Text size (0.3=small, 0.5=default, 0.8=medium, 1.2=large, 1.5=extra large)
TEXT_FONT_THICKNESS = 1  # Text thickness
TEXT_FONT_COLOR = (0, 0, 0)  # BGR: Black text
TEXT_LINE_HEIGHT = 25  # Space between lines

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
