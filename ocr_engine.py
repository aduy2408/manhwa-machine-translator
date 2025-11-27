"""OCR engine using PaddleOCR Korean TextRecognition model."""

import cv2
import numpy as np
import tempfile
import os
import traceback
import time
from paddleocr import TextRecognition
from typing import Optional, List


class OCREngine:
    """Wrapper around PaddleOCR TextRecognition for Korean text recognition."""

    def __init__(self):
        """Initialize the OCR engine with Korean TextRecognition model."""
        try:
            print("Initializing OCR Engine with korean_PP-OCRv5_mobile_rec model...")
            self.model = TextRecognition(
                model_name="korean_PP-OCRv5_mobile_rec"
            )
            self.last_raw_results = None
            print("OCR Engine initialized successfully")
        except Exception as e:
            print(f"Failed to initialize OCR Engine: {str(e)}")
            traceback.print_exc()
            raise

    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text from an image.

        Args:
            image: OpenCV image (BGR format)

        Returns:
            Recognized text as a single string
        """
        if image is None or image.size == 0:
            print("Error: Empty or None image provided")
            return ""

        try:
            print(f"Starting OCR on image with shape: {image.shape}")
            
            # Create temp directory if it doesn't exist
            os.makedirs("./temp", exist_ok=True)
            
            # Save image to temporary file (PaddleOCR expects file path)
            tmp_path = f"./temp/ocr_input_{os.getpid()}_{int(time.time() * 1000)}.png"
            cv2.imwrite(tmp_path, image)
            print(f"Saved image to: {tmp_path}")

            try:
                # Run text recognition
                print("Running text recognition...")
                results = self.model.predict(input=tmp_path, batch_size=1)
                print(f"OCR returned: {results}")
                print(f"OCR returned type: {type(results)}")
                self.last_raw_results = results

                # Extract text from results
                recognized_text = ""
                if results:
                    if isinstance(results, list):
                        for i, res in enumerate(results):
                            print(f"Result {i}: {res}")
                            print(f"Result type: {type(res)}")
                            if isinstance(res, dict):
                                if "rec_text" in res:
                                    recognized_text += res["rec_text"] + "\n"
                                elif "text" in res:
                                    recognized_text += res["text"] + "\n"
                            elif hasattr(res, "text"):
                                recognized_text += res.text + "\n"
                            elif isinstance(res, str):
                                recognized_text += res + "\n"
                            else:
                                recognized_text += str(res) + "\n"
                    elif isinstance(results, dict):
                        print(f"Results is dict: {results.keys()}")
                        if "rec_text" in results:
                            recognized_text = results["rec_text"]
                        elif "text" in results:
                            recognized_text = results["text"]
                        else:
                            recognized_text = str(results)
                    else:
                        recognized_text = str(results)

                print(f"Extracted text: {repr(recognized_text)}")
                return recognized_text.strip()
            finally:
                # Keep the file for debugging (don't delete)
                print(f"OCR input image saved at: {os.path.abspath(tmp_path)}")

        except Exception as e:
            print(f"OCR Error: {str(e)}")
            traceback.print_exc()
            return ""

    def get_confidence_scores(self) -> List[float]:
        """Get confidence scores from the last recognition."""
        if self.last_raw_results is None:
            return []

        scores = []
        if isinstance(self.last_raw_results, list):
            for res in self.last_raw_results:
                if isinstance(res, dict) and "rec_score" in res:
                    scores.append(res["rec_score"])
                elif hasattr(res, "confidence"):
                    scores.append(res.confidence)
        elif isinstance(self.last_raw_results, dict) and "rec_score" in self.last_raw_results:
            scores.append(self.last_raw_results["rec_score"])

        return scores

    def get_average_confidence(self) -> float:
        """Get average confidence score from the last recognition."""
        scores = self.get_confidence_scores()
        return sum(scores) / len(scores) if scores else 0.0
