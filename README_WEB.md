# Korean Manhwa Translator - Web Version

A modern web-based application for detecting and translating Korean text in manhwa (webtoon) image

## USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 
## USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 
## USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 



## YOU CAN ALSO TRY TO USE A DIFFERENT OCR MODEL(THIS REPO USES KOREAN PADDLE OCR), JUST CHANGE THE MODEL NAME FOR RECOGNIZING TEXT, LIKE JAPANESE, IF PADDLE HAS THAT MODEL
## YOU CAN ALSO TRY TO USE A DIFFERENT OCR MODEL(THIS REPO USES KOREAN PADDLE OCR), JUST CHANGE THE MODEL NAME FOR RECOGNIZING TEXT, LIKE JAPANESE, IF PADDLE HAS THAT MODEL
## Overview

The web version provides a complete solution for Korean text extraction and translation with:
- **Single and batch image processing**
- **Real-time translation with AI**
- **Responsive web interface**
- **REST API backend**
- **Support for folder processing**

## System Requirements


## Project Structure

```
.
├── app.py                      # FastAPI backend server
├── image_processor.py          # Image processing utilities
├── ocr_engine.py               # PaddleOCR wrapper
├── translator.py               # Google Gemini API integration
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── index.html                  # Static HTML file
├── web/                        # React frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── index.js
│   │   ├── App.js
│   │   ├── App.css
│   │   └── components/
│   └── package.json
└── README_WEB.md               # This file
```

## Installation & Setup

### Backend Setup

#### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure Environment Variables


Edit `.env` and add your Google Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```


#### 4. Start the Backend Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Access

The web interface is automatically served by the FastAPI backend. Once the backend is running, access the application at:

```
http://localhost:5000
```


Detects and recognizes Korean text in a single image.

**Response:**
```json
{
  "success": true,
  "recognized_texts": [
    {
      "text": "안녕하세요",
      "confidence": 0.98
    }
  ],
  "polygons": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
  "image_shape": [height, width, channels]
}
```

### Translate Texts
```
POST /translate
Content-Type: application/json

{
  "texts": ["안녕하세요", "감사합니다"],
  "target_language": "English"
}
```

Translates multiple Korean texts to the specified language.

**Response:**
```json
{
  "success": true,
  "translations": {
    "안녕하세요": "Hello",
    "감사합니다": "Thank you"
  }
}
```

### Process Image (Full Pipeline)
```
POST /process
Content-Type: multipart/form-data

Form Data:
- file: <image file>
```

Complete pipeline: detect → translate → draw translations on image.

**Response:**
```json
{
  "success": true,
  "detected_texts": [
    {
      "text": "안녕하세요",
      "confidence": 0.98
    }
  ],
  "translations": {
    "안녕하세요": "Hello"
  },
  "polygons": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
  "image": "data:image/png;base64,iVBORw0KG...",
  "image_shape": [height, width, channels]
}
```

### Batch Process Images
```
POST /batch-process
Content-Type: multipart/form-data

Form Data:
- files: <multiple image files>
```

Processes multiple images at once, extracting and translating all unique text with context consistency.

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "detected_texts": [...],
      "polygons": [...],
      "image": "data:image/png;base64,..."
    }
  ],
  "translations": {
    "안녕하세요": "Hello",
    ...
  }
}
```

### Process Folder
```
POST /process-folder
Content-Type: application/json

{
  "folder_path": "/path/to/folder"
}
```

Processes all images in a folder sequentially with consistent translation context.

**Response:**
```json
{
  "success": true,
  "results": [...],
  "translations": {...}
}
```

## Features

### Single Image Processing
- Upload a single image
- Automatic text detection
- Real-time translation
- Visual preview of results

### Batch Processing
- Upload multiple images at once
- Consistent translation across batch
- Maintains context and terminology
- Organized output

### Folder Processing
- Process entire folders automatically
- Natural sorting of image files
- Context-aware translation
- Export results

### REST API
- Language-agnostic backend
- JSON responses
- Base64 encoded images
- Comprehensive error handling

## Configuration

Edit `config.py` to customize application behavior:

```python

# Translation Settings
GEMINI_MODEL = "gemini-2.5-flash"

# Image Processing
MAX_IMAGE_SIZE = 2000

# API Settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
```
