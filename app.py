"""FastAPI backend for Korean text translation app."""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2
import base64
from typing import List, Dict, Optional
import io

from translator import Translator
from gui.text_detector import TextDetector
from image_processor import ImageProcessor

app = FastAPI(
    title="Korean Manhwa Translator",
    description="Detect and translate Korean text in manhwa images",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
translator = Translator()
text_detector = TextDetector()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}


# Serve index.html
@app.get("/")
async def root():
    return FileResponse("web/public/index.html")


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 for JSON response."""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/detect")
async def detect_text(file: UploadFile = File(...)):
    """Detect and recognize text in uploaded image."""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Read image
        file_bytes = await file.read()
        file_bytes_np = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        
        # Detect text
        result = text_detector.detect_and_recognize_text(image)
        if not result:
            raise HTTPException(status_code=400, detail="No text detected")
        
        polygons, recognized_texts = result
        
        # Convert polygons to list for JSON serialization
        polygons_list = [poly.tolist() if isinstance(poly, np.ndarray) else poly for poly in polygons]
        
        return {
            'success': True,
            'recognized_texts': recognized_texts,
            'polygons': polygons_list,
            'image_shape': list(image.shape)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate_texts(texts: List[str], target_language: str = "English"):
    """Translate multiple Korean texts."""
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        translations = translator.translate_batch(texts, target_language)
        return {
            'success': True,
            'translations': translations
        }
    except Exception as e:
        print(f"[API] Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_image(file: UploadFile = File(...), context: str = None):
    """Full pipeline: detect -> translate -> draw on image."""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Read image
        file_bytes = await file.read()
        file_bytes_np = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        
        # Step 1: Detect text
        result = text_detector.detect_and_recognize_text(image)
        if not result:
            return {
                'detected_texts': [],
                'image': encode_image(image)
            }
        
        polygons, recognized_texts = result
        
        # Step 2: Extract unique texts for translation
        korean_texts = [item['text'] for item in recognized_texts]
        unique_texts = list(dict.fromkeys(korean_texts))
        
        # Step 3: Translate with context if provided
        translations = translator.translate_batch(unique_texts, context=context)
        
        if not translations:
            raise HTTPException(status_code=500, detail="Translation failed")
        
        # Step 4: Draw on image
        image_with_translations = image.copy()
        ImageProcessor.draw_translations_on_image(
            image_with_translations, polygons, recognized_texts, translations
        )
        
        # Convert polygons to list for JSON
        polygons_list = [poly.tolist() if isinstance(poly, np.ndarray) else poly for poly in polygons]
        
        return {
            'success': True,
            'detected_texts': recognized_texts,
            'translations': translations,
            'polygons': polygons_list,
            'image': encode_image(image_with_translations),
            'image_shape': list(image.shape)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-process")
async def batch_process(files: List[UploadFile] = File(...), context: str = None):
    """Process multiple images at once."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        from natsort import natsorted
        
        results = []
        all_translations = {}
        
        # Step 1: Detect text in all images
        images = []
        filenames = []
        for file in files:
            if not allowed_file(file.filename):
                continue
            file_bytes = await file.read()
            file_bytes_np = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
                filenames.append(file.filename)
        
        # Sort by filename using natsort
        sorted_filenames = natsorted(filenames)
        sorted_images = [images[filenames.index(f)] for f in sorted_filenames]
        images = sorted_images
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images")
        
        # Detect text in all images
        all_texts = []
        detection_results = []
        for image in images:
            result = text_detector.detect_and_recognize_text(image)
            if result:
                polygons, recognized_texts = result
                detection_results.append((image, polygons, recognized_texts))
                for item in recognized_texts:
                    korean_text = item['text']
                    if korean_text not in all_texts:
                        all_texts.append(korean_text)
        
        # Step 2: Batch translate with context if provided
        if all_texts:
            all_translations = translator.translate_batch(all_texts, context=context)
        
        # Step 3: Draw on images
        for image, polygons, recognized_texts in detection_results:
            image_with_translations = image.copy()
            ImageProcessor.draw_translations_on_image(
                image_with_translations, polygons, recognized_texts, all_translations
            )
            
            polygons_list = [poly.tolist() if isinstance(poly, np.ndarray) else poly for poly in polygons]
            results.append({
                'detected_texts': recognized_texts,
                'polygons': polygons_list,
                'image': encode_image(image_with_translations),
                'image_shape': list(image.shape)
            })
        
        return {
            'success': True,
            'results': results,
            'translations': all_translations
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-folder")
async def process_folder(folder_path: str):
    """Process all images in a folder."""
    import os
    from pathlib import Path
    from natsort import natsorted
    
    try:
        folder = Path(folder_path)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail="Invalid folder path")
        
        # Get all image files
        image_files = []
        for ext in ALLOWED_EXTENSIONS:
            image_files.extend(folder.glob(f'*.{ext}'))
            image_files.extend(folder.glob(f'*.{ext.upper()}'))
        
        if not image_files:
            raise HTTPException(status_code=400, detail="No images found in folder")
        
        # Natural sort to handle _0, _1, _2 etc. correctly
        image_files = natsorted(set(image_files))
        
        results = []
        all_translations = {}
        
        # Step 1: Detect text in all images
        all_texts = []
        detection_results = []
        for image_path in image_files:
            print(f"[API] Processing: {image_path.name}")
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            result = text_detector.detect_and_recognize_text(image)
            if result:
                polygons, recognized_texts = result
                detection_results.append((image, polygons, recognized_texts))
                for item in recognized_texts:
                    korean_text = item['text']
                    if korean_text not in all_texts:
                        all_texts.append(korean_text)
        
        if not detection_results:
            return {
                'success': True,
                'results': [],
                'translations': {},
                'message': 'No text detected in any images'
            }
        
        # Step 2: Batch translate
        if all_texts:
            all_translations = translator.translate_batch(all_texts)
        
        # Step 3: Draw on images
        for image, polygons, recognized_texts in detection_results:
            image_with_translations = image.copy()
            ImageProcessor.draw_translations_on_image(
                image_with_translations, polygons, recognized_texts, all_translations
            )
            
            polygons_list = [poly.tolist() if isinstance(poly, np.ndarray) else poly for poly in polygons]
            results.append({
                'detected_texts': recognized_texts,
                'polygons': polygons_list,
                'image': encode_image(image_with_translations),
                'image_shape': list(image.shape)
            })
        
        return {
            'success': True,
            'results': results,
            'translations': all_translations
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Folder processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
