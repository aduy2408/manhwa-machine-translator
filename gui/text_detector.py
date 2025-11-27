"""Text detection and recognition using PaddleOCR."""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import sys
import os
import cv2

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DETECTION_CONFIDENCE_THRESHOLD, RECOGNITION_CONFIDENCE_THRESHOLD


class TextDetector:
    """Detects and recognizes text regions in images using PaddleOCR."""

    def __init__(self):
        """Initialize the text detection and recognition models."""
        try:
            from paddleocr import TextDetection, TextRecognition
            self.det_model = TextDetection(model_name="PP-OCRv5_mobile_det")
            self.rec_model = TextRecognition(model_name="korean_PP-OCRv5_mobile_rec")
            print("[TextDetector] Models loaded successfully")
        except Exception as e:
            print(f"[TextDetector] Error loading models: {e}")
            self.det_model = None
            self.rec_model = None

    def detect_and_recognize_text(
        self, image, detection_threshold: float = None, recognition_threshold: float = None
    ) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """
        Detect text regions and recognize text in an image.

        Args:
            image: Image as numpy array (BGR format from cv2)
            detection_threshold: Filter detection bboxes with dt_scores > threshold
                                (defaults to DETECTION_CONFIDENCE_THRESHOLD from config)
            recognition_threshold: Filter recognized text with rec_score > threshold
                                  (defaults to RECOGNITION_CONFIDENCE_THRESHOLD from config)

        Returns:
            Tuple of (detected polygons, list of dicts with text and confidence) or None
        """
        if detection_threshold is None:
            detection_threshold = DETECTION_CONFIDENCE_THRESHOLD
        if recognition_threshold is None:
            recognition_threshold = RECOGNITION_CONFIDENCE_THRESHOLD
            
        if self.det_model is None or self.rec_model is None:
            print("[TextDetector] Models not loaded")
            return None

        try:
            # Detect text regions
            det_result = self.det_model.predict([image], batch_size=1)
            if not det_result or len(det_result) == 0:
                return None

            dt_polys = det_result[0].get("dt_polys")
            dt_scores = det_result[0].get("dt_scores")
            
            if dt_polys is None or len(dt_polys) == 0:
                print("[TextDetector] No text regions detected")
                return None

            # Ensure dt_polys is list-like
            if isinstance(dt_polys, np.ndarray):
                dt_polys = list(dt_polys)
            
            print(f"[TextDetector] Detected {len(dt_polys)} text regions")

            # Recognize text in each detected region
            recognized_texts = []
            valid_polys = []
            kept_count = 0

            for idx, poly in enumerate(dt_polys):
                # Ensure poly is numpy array for indexing
                if not isinstance(poly, np.ndarray):
                    poly = np.array(poly)
                
                # Check detection confidence first
                det_confidence = 1.0
                if dt_scores is not None and idx < len(dt_scores):
                    try:
                        det_confidence = float(dt_scores[idx])
                    except (IndexError, TypeError, ValueError):
                        det_confidence = 1.0
                
                if det_confidence <= detection_threshold:
                    print(
                        f"[TextDetector] Skipped region {idx}: detection confidence {det_confidence:.3f} <= {detection_threshold}"
                    )
                    continue
                
                # Get bounding box from polygon (convert to int)
                xs = poly[:, 0].astype(int)
                ys = poly[:, 1].astype(int)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())

                # Crop region
                cropped = image[y_min:y_max, x_min:x_max]
                print(f"[TextDetector] Region {idx}: crop bounds x=[{x_min}:{x_max}], y=[{y_min}:{y_max}], size={cropped.shape}")
                
                # Preprocess cropped region for better recognition
                try:
                    # Convert to grayscale if needed
                    if len(cropped.shape) == 3 and cropped.shape[2] == 3:
                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = cropped if len(cropped.shape) == 2 else cropped[:, :, 0]
                    
                    # Enhance contrast using CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray)
                    
                    # Convert back to BGR
                    cropped_processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    print(f"[TextDetector] Warning: preprocessing failed, using original: {e}")
                    cropped_processed = cropped

                # Recognize text
                try:
                    rec_result = self.rec_model.predict([cropped_processed], batch_size=1)
                    if rec_result and len(rec_result) > 0:
                        text = rec_result[0].get("rec_text", "")
                        rec_confidence = rec_result[0].get("rec_score", 0.0)

                        # Only keep if recognition confidence above threshold
                        if rec_confidence > recognition_threshold:
                            recognized_texts.append({"text": text, "confidence": float(rec_confidence)})
                            valid_polys.append(poly)
                            kept_count += 1
                            print(
                                f"[TextDetector] Recognized: '{text}' (det: {det_confidence:.3f}, rec: {rec_confidence:.3f})"
                            )
                        else:
                            print(
                                f"[TextDetector] Skipped: '{text}' (rec: {rec_confidence:.3f}) - below threshold {recognition_threshold}"
                            )
                    else:
                        print(f"[TextDetector] No text recognized in region {idx}")
                except Exception as e:
                    print(f"[TextDetector] Error recognizing text: {e}")

            print(
                f"[TextDetector] Kept {kept_count}/{len(dt_polys)} regions (detection>{detection_threshold}, recognition>{recognition_threshold})"
            )

            if len(valid_polys) == 0:
                return None

            # Return only valid polygons and their recognized texts
            return np.array(valid_polys), recognized_texts

        except Exception as e:
            print(f"[TextDetector] Error during detection: {e}")
            return None

    def detect_text(self, image) -> Optional[np.ndarray]:
        """
        Detect text regions in an image (legacy method).

        Args:
            image: Image as numpy array (BGR format from cv2)

        Returns:
            Detected polygons as numpy array or None if detection fails
        """
        result = self.detect_and_recognize_text(image)
        if result:
            return result[0]
        return None

    def detect_batch(
        self, images: List, detection_threshold: float = None, recognition_threshold: float = None
    ) -> Dict[int, Optional[Tuple[np.ndarray, List[Dict]]]]:
        """
        Detect and recognize text in multiple images.

        Args:
            images: List of images as numpy arrays
            detection_threshold: Filter detection bboxes with dt_scores > threshold
                                (defaults to DETECTION_CONFIDENCE_THRESHOLD from config)
            recognition_threshold: Filter recognized text with rec_score > threshold
                                  (defaults to RECOGNITION_CONFIDENCE_THRESHOLD from config)

        Returns:
            Dictionary with image index -> (detected polygons, recognized texts with confidence)
        """
        if detection_threshold is None:
            detection_threshold = DETECTION_CONFIDENCE_THRESHOLD
        if recognition_threshold is None:
            recognition_threshold = RECOGNITION_CONFIDENCE_THRESHOLD
            
        results = {}
        for idx, image in enumerate(images):
            print(
                f"[TextDetector] Detecting/recognizing text in image {idx + 1}/{len(images)}"
            )
            results[idx] = self.detect_and_recognize_text(image, detection_threshold, recognition_threshold)
        return results

    @staticmethod
    def polygons_to_bboxes(polygons: np.ndarray) -> List[List[int]]:
        """
        Convert detected polygons to bounding boxes.

        Args:
            polygons: Detected polygon points

        Returns:
            List of [x, y, w, h] bounding boxes
        """
        bboxes = []
        if polygons is None:
            return bboxes

        for poly in polygons:
            # Get min/max x and y from polygon points
            xs = poly[:, 0]
            ys = poly[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            w = x_max - x_min
            h = y_max - y_min
            bboxes.append([x_min, y_min, w, h])

        return bboxes

    @staticmethod
    def save_detection_json(
        detections: Dict[int, Optional[Tuple[np.ndarray, List[Dict]]]], save_path: str
    ) -> bool:
        """
        Save detected text regions and recognized text as JSON.

        Args:
            detections: Dictionary with image index -> (polygons, texts with confidence)
            save_path: Path to save JSON file

        Returns:
            True if successful
        """
        try:
            data = {}
            for idx, result in detections.items():
                img_key = f"pic_{idx + 1}"
                if result is not None:
                    polys, texts = result
                    bboxes = TextDetector.polygons_to_bboxes(polys)
                    data[img_key] = {"bbox": bboxes, "text": texts}
                else:
                    data[img_key] = {"bbox": [], "text": []}

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[TextDetector] Detection results saved to {save_path}")
            return True
        except Exception as e:
            print(f"[TextDetector] Error saving JSON: {e}")
            return False
