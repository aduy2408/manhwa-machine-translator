"""Worker threads for OCR and translation processing."""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np


class OCRWorker(QThread):
    """Worker thread for OCR processing."""

    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)

    def __init__(self, region_image, ocr_engine):
        super().__init__()
        self.region_image = region_image
        self.ocr_engine = ocr_engine

    def run(self):
        try:
            print(f"[OCRWorker] Starting with image shape: {self.region_image.shape}")
            text = self.ocr_engine.recognize_text(self.region_image)
            print(f"[OCRWorker] OCR completed. Text: {repr(text)}")
            confidence = self.ocr_engine.get_average_confidence()
            print(f"[OCRWorker] Confidence: {confidence}")
            self.finished.emit(text, confidence)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"[OCRWorker] Error: {error_msg}")
            self.error.emit(error_msg)


class AutoTranslateWorker(QThread):
    """Worker thread for auto-detect and translate."""

    finished = pyqtSignal(object, object, object)  # image, recognized_texts, translations
    error = pyqtSignal(str)

    def __init__(self, image, text_detector, translator):
        super().__init__()
        self.image = image
        self.text_detector = text_detector
        self.translator = translator

    def run(self):
        try:
            # Step 1: Detect text
            detection_result = self.text_detector.detect_and_recognize_text(self.image)
            if not detection_result:
                self.error.emit("No text detected in image.")
                return

            polygons, recognized_texts = detection_result

            # Step 2: Extract Korean texts
            korean_texts = [item["text"] for item in recognized_texts]
            print(f"[AutoTranslateWorker] Detected texts: {korean_texts}")

            # Step 3: Batch translate
            translations = self.translator.translate_batch(korean_texts)
            print(f"[AutoTranslateWorker] Translations: {translations}")

            if not translations:
                self.error.emit("Translation failed.")
                return

            # Return results
            self.finished.emit(polygons, recognized_texts, translations)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"[AutoTranslateWorker] Error: {error_msg}")
            self.error.emit(error_msg)
