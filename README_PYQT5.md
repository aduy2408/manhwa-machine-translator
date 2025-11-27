# Korean Manhwa Translator - PyQt5 Version

A desktop application for extracting and translating Korean text from manhwa (webtoon) images using PyQt5, PaddleOCR, and Google Gemini API.

# USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 
# USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 
# USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 
# USE IMAGE ASSISTANT DOWNLOADER TO INSTALL ALL IMAGES OF A MANHWA CHAPTER FRON WFWF 


# YOU CAN ALSO TRY TO USE A DIFFERENT OCR MODEL(THIS REPO USES KOREAN PADDLE OCR), JUST CHANGE THE MODEL NAME FOR RECOGNIZING TEXT, LIKE JAPANESE, IF PADDLE HAS THAT MODEL
# YOU CAN ALSO TRY TO USE A DIFFERENT OCR MODEL(THIS REPO USES KOREAN PADDLE OCR), JUST CHANGE THE MODEL NAME FOR RECOGNIZING TEXT, LIKE JAPANESE, IF PADDLE HAS THAT MODEL

## Features

- **Image Import**: Load images directly from your system
- **Interactive Selection**: Use the paint brush tool to select specific areas containing Korean text
- **Korean OCR**: Extract Korean text using PaddleOCR with the korean_PP-OCRv5_mobile_rec model
- **AI Translation**: Translate recognized Korean text to English using Google Gemini API
- **Confidence Scoring**: View OCR confidence levels for each recognized text

## Installation


### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Google Gemini API

Edit `.env` and add your API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Start the Application

```bash
python main.py
```

## Configuration

Edit `config.py` to customize application behavior:

```python
# OCR Settings

# UI Settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
DEFAULT_BRUSH_SIZE = 5

# Model Settings
GEMINI_MODEL = "gemini-2.5-flash"
```

## Project Structure

```
.
├── main.py                  # Application entry point
├── gui/
│   ├── __init__.py
│   ├── main_window.py       # Main application window
│   ├── canvas.py            # Image canvas with brush tool
│   ├── workers.py           # Threading workers for OCR & translation
│   ├── scroll_manager.py     # Scroll handling
│   └── text_detector.py      # Text detection integration
├── image_processor.py        # Image handling and processing
├── ocr_engine.py             # PaddleOCR wrapper
├── translator.py             # Google Gemini API integration
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README_PYQT5.md          # This file
```

## Module Reference

### `main.py`
Entry point for the application. Initializes QApplication and MainWindow.

### `gui/main_window.py`
- `MainWindow`: Main application window with menu, toolbar, and layout
- Manages image loading, brush tools, and result display
- Handles threading for non-blocking operations

### `gui/canvas.py`
- `ImageCanvas`: Custom QLabel for displaying images
- Supports mouse events for brush drawing
- Real-time brush preview

### `gui/workers.py`
- `OCRWorker`: QThread worker for non-blocking OCR processing
- `TranslationWorker`: QThread worker for non-blocking translation
- Emits signals for UI updates

### `image_processor.py`
- Image loading and resizing
- Brush mask creation and region extraction
- Image processing utilities

### `ocr_engine.py`
- `OCREngine`: Wrapper around PaddleOCR
- Handles Korean text recognition
- Returns text positions and confidence scores

### `translator.py`
- `Translator`: Google Gemini API integration
- Batch translation support
- Error handling and retry logic

### `config.py`
Centralized configuration for all modules without code modification.
