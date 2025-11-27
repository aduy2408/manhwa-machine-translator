"""Image processing utilities for selection and extraction."""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, List, Dict
from natsort import natsorted
from config import (
    RECTANGLE_PADDING,
    MAX_IMAGE_SIZE,
    TEXT_FONT_SCALE,
    TEXT_FONT_THICKNESS,
    TEXT_FONT_COLOR,
    TEXT_LINE_HEIGHT,
)


class ImageProcessor:
    """Handles image loading, manipulation, and region extraction."""

    def __init__(self):
        self.original_image = None
        self.display_image = None
        self.mask = None
        self.scale_factor = 1.0
        self.last_selection_bbox = None  # Store bounding box of last selection
        self.image_list = []  # For folder loading: list of separate images
        self.current_image_index = 0  # Current image being displayed from list

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file and resize if necessary."""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Resize if too large
        height, width = self.original_image.shape[:2]
        if max(height, width) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(height, width)
            self.scale_factor = scale
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(
                self.original_image, (new_width, new_height)
            )
        else:
            self.scale_factor = 1.0

        self.display_image = self.original_image.copy()
        self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
        return self.display_image

    def load_folder(self, folder_path: str) -> np.ndarray:
        """Load all images from folder, sort naturally, display one by one."""
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        
        # Get all image files
        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in supported_formats
        ]
        
        if not image_files:
            raise ValueError(f"No images found in folder: {folder_path}")
        
        # Sort naturally (0, 1, 2, 10, 11 instead of 0, 1, 10, 11, 2)
        image_files = natsorted(image_files)
        print(f"[ImageProcessor] Found {len(image_files)} images")
        
        # Load all images
        self.image_list = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ImageProcessor] Failed to load {img_path}, skipping")
                continue
            
            # Resize if too large
            height, width = img.shape[:2]
            if max(height, width) > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            self.image_list.append(img)
        
        if not self.image_list:
            raise ValueError("Failed to load any images")
        
        # Start with first image (use reference, not copy)
        self.current_image_index = 0
        self.original_image = self.image_list[0]
        self.display_image = self.original_image.copy()
        self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
        self.scale_factor = 1.0
        
        print(f"[ImageProcessor] Loaded {len(self.image_list)} images. Displaying image 1/{len(self.image_list)}")
        return self.display_image
    
    def next_image(self) -> bool:
        """Go to next image in the list. Returns True if there's a next image."""
        if not self.image_list or self.current_image_index >= len(self.image_list) - 1:
            return False
        
        self.current_image_index += 1
        # Use reference to the actual image in list (not a copy)
        self.original_image = self.image_list[self.current_image_index]
        self.display_image = self.original_image.copy()
        self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
        
        print(f"[ImageProcessor] Now displaying image {self.current_image_index + 1}/{len(self.image_list)}")
        return True
    
    def prev_image(self) -> bool:
        """Go to previous image in the list. Returns True if there's a previous image."""
        if not self.image_list or self.current_image_index <= 0:
            return False
        
        self.current_image_index -= 1
        # Use reference to the actual image in list (not a copy)
        self.original_image = self.image_list[self.current_image_index]
        self.display_image = self.original_image.copy()
        self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
        
        print(f"[ImageProcessor] Now displaying image {self.current_image_index + 1}/{len(self.image_list)}")
        return True
    
    def get_image_info(self) -> str:
        """Get info about current image in list."""
        if not self.image_list:
            return "No images loaded"
        return f"Image {self.current_image_index + 1}/{len(self.image_list)}"

    def draw_brush_stroke(self, x: int, y: int, size: int = 3, color: Tuple = (0, 255, 0)):
        """Draw a brush stroke on the mask at the given coordinates."""
        if self.mask is None:
            return

        # Draw on mask
        cv2.circle(self.mask, (x, y), size, 255, -1)

        # Draw on display image
        cv2.circle(self.display_image, (x, y), size, color, -1)

    def get_selected_region(self) -> Optional[np.ndarray]:
        """Extract the rectangular region containing the painted area."""
        if self.mask is None or self.mask.sum() == 0:
            print("[ImageProcessor] No mask data - no selection")
            return None

        print(f"[ImageProcessor] Mask sum: {self.mask.sum()}, Mask shape: {self.mask.shape}")

        # Find contours of painted area
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[ImageProcessor] Found {len(contours)} contours")
        
        if not contours:
            print("[ImageProcessor] No contours found")
            return None

        # Get bounding rectangle
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        print(f"[ImageProcessor] Bounding rect: x={x}, y={y}, w={w}, h={h}")

        # Add padding and ensure it's a rectangle/square
        x = max(0, x - RECTANGLE_PADDING)
        y = max(0, y - RECTANGLE_PADDING)
        w = min(self.original_image.shape[1] - x, w + 2 * RECTANGLE_PADDING)
        h = min(self.original_image.shape[0] - y, h + 2 * RECTANGLE_PADDING)

        print(f"[ImageProcessor] Final region: x={x}, y={y}, w={w}, h={h}")
        print(f"[ImageProcessor] Original image shape: {self.original_image.shape}")

        # Store bounding box for later use
        self.last_selection_bbox = (x, y, w, h)
        
        # Extract and return the region
        region = self.original_image[y : y + h, x : x + w]
        print(f"[ImageProcessor] Extracted region shape: {region.shape}")
        return region

    def reset_mask(self):
        """Reset the drawing mask."""
        if self.display_image is not None:
            self.display_image = self.original_image.copy()
            self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)

    def get_display_image(self) -> Optional[np.ndarray]:
        """Get the current display image with brush strokes."""
        return self.display_image

    def replace_region_with_text(self, translated_text: str) -> bool:
        """
        Replace the last selected region with a white box and translated text.
        Edits are permanent - saved to both display and original image.
        
        Args:
            translated_text: Text to write in the region
            
        Returns:
            True if successful, False otherwise
        """
        if self.last_selection_bbox is None or self.original_image is None:
            print("[ImageProcessor] No selection bbox or image")
            return False

        return self.draw_text_on_images(
            self.original_image, self.display_image, self.last_selection_bbox, translated_text
        )

    @staticmethod
    def draw_text_on_images(
        original_img: np.ndarray,
        display_img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        text: str,
    ) -> bool:
        """
        Draw white rectangle and text on both original and display images.
        
        Args:
            original_img: Original image to modify
            display_img: Display image to modify
            bbox: Bounding box (x, y, w, h)
            text: Text to write
            
        Returns:
            True if successful, False otherwise
        """
        x, y, w, h = bbox
        print(f"[ImageProcessor] Drawing text on region at x={x}, y={y}, w={w}, h={h}")

        # Create white rectangle on BOTH images (permanent)
        original_img[y : y + h, x : x + w] = [255, 255, 255]
        display_img[y : y + h, x : x + w] = [255, 255, 255]

        # Write text on the white rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = TEXT_FONT_SCALE
        font_color = TEXT_FONT_COLOR
        thickness = TEXT_FONT_THICKNESS

        # Split text into lines if it's too long
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

            if text_size[0] > w - 10:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)

        # Draw text lines - centered vertically
        line_height = TEXT_LINE_HEIGHT
        total_text_height = len(lines) * line_height

        # Calculate starting y position to center text vertically
        y_offset = y + (h - total_text_height) // 2 + line_height // 2

        for line in lines:
            if y_offset + line_height > y + h:
                break
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x_centered = x + (w - text_size[0]) // 2

            # Draw text on BOTH images (permanent)
            cv2.putText(
                display_img,
                line,
                (x_centered, y_offset),
                font,
                font_scale,
                font_color,
                thickness,
            )
            cv2.putText(
                original_img,
                line,
                (x_centered, y_offset),
                font,
                font_scale,
                font_color,
                thickness,
            )
            y_offset += line_height

        print(f"[ImageProcessor] Text drawing completed")
        return True

    @staticmethod
    def _resolve_bbox_overlaps(bboxes: List[Tuple[int, int, int, int]], img_h: int, img_w: int) -> List[Tuple[int, int, int, int]]:
        """
        Resolve overlapping bounding boxes with minimal movement.
        """
        if not bboxes:
            return bboxes
        
        resolved = list(bboxes)
        min_gap = 1
        small_move = 3  # Only move 3 pixels to avoid overlap
        
        for i in range(len(resolved)):
            x1_min, y1_min, x1_max, y1_max = resolved[i]
            
            for j in range(i + 1, len(resolved)):
                x2_min, y2_min, x2_max, y2_max = resolved[j]
                
                # Check if boxes overlap
                if not (x1_max + min_gap < x2_min or x2_max + min_gap < x1_min or 
                        y1_max + min_gap < y2_min or y2_max + min_gap < y1_min):
                    
                    w2 = x2_max - x2_min
                    h2 = y2_max - y2_min
                    
                    # Move box j down by small amount
                    new_y2_min = y1_max + small_move
                    if new_y2_min + h2 <= img_h:
                        resolved[j] = (x2_min, new_y2_min, x2_max, new_y2_min + h2)
        
        return resolved

    @staticmethod
    def draw_translations_on_image(
        image: np.ndarray,
        polygons: np.ndarray,
        recognized_texts: list,
        translations: dict,
    ) -> bool:
        """
        Draw translations on detected text regions.
        
        Args:
            image: Image to modify (will be modified in-place)
            polygons: Detected polygons from text detection
            recognized_texts: List of dicts with 'text' and 'confidence' keys
            translations: Dictionary mapping original Korean text -> translated text
            
        Returns:
            True if successful
        """
        if polygons is None or len(polygons) == 0:
            return False
        
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = TEXT_FONT_SCALE
            font_color = TEXT_FONT_COLOR
            thickness = TEXT_FONT_THICKNESS
            
            print(f"[ImageProcessor] Starting to draw translations on {len(polygons)} regions")
            print(f"[ImageProcessor] Image shape: {image.shape}")
            print(f"[ImageProcessor] Recognized texts: {len(recognized_texts)}")
            
            # First pass: calculate all bounding boxes
            bboxes = []
            img_h, img_w = image.shape[:2]
            
            for idx, poly in enumerate(polygons):
                # Ensure poly is numpy array and convert to float first, then int
                if not isinstance(poly, np.ndarray):
                    poly = np.array(poly, dtype=np.float32)
                else:
                    poly = poly.astype(np.float32)
                
                # Get bounding box from polygon - round to nearest integer
                xs = np.round(poly[:, 0]).astype(int)
                ys = np.round(poly[:, 1]).astype(int)
                x_min = int(np.min(xs))
                x_max = int(np.max(xs))
                y_min = int(np.min(ys))
                y_max = int(np.max(ys))
                
                # Clamp to image bounds
                x_min = max(0, min(x_min, img_w - 1))
                x_max = max(x_min + 1, min(x_max, img_w))
                y_min = max(0, min(y_min, img_h - 1))
                y_max = max(y_min + 1, min(y_max, img_h))
                
                bboxes.append((x_min, y_min, x_max, y_max))
            
            for idx, poly in enumerate(polygons):
                # Ensure poly is numpy array and convert to float first, then int
                if not isinstance(poly, np.ndarray):
                    poly = np.array(poly, dtype=np.float32)
                else:
                    poly = poly.astype(np.float32)
                
                # Get bounding box from polygon - round to nearest integer
                xs = np.round(poly[:, 0]).astype(int)
                ys = np.round(poly[:, 1]).astype(int)
                x_min = int(np.min(xs))
                x_max = int(np.max(xs))
                y_min = int(np.min(ys))
                y_max = int(np.max(ys))
                
                # Clamp to image bounds
                img_h, img_w = image.shape[:2]
                x_min = max(0, min(x_min, img_w - 1))
                x_max = max(x_min + 1, min(x_max, img_w))
                y_min = max(0, min(y_min, img_h - 1))
                y_max = max(y_min + 1, min(y_max, img_h))
                
                w = x_max - x_min
                h = y_max - y_min
                
                print(f"[ImageProcessor] Region {idx}: resolved bbox=({x_min},{y_min},{w},{h}), image shape={image.shape}")
                
                # Fill with white
                image[y_min:y_max, x_min:x_max] = [255, 255, 255]
                
                # Get the Korean text for this polygon
                translated_text = ""
                korean_text = ""
                if idx < len(recognized_texts):
                    text_item = recognized_texts[idx]
                    # Handle both dict and string formats
                    if isinstance(text_item, dict):
                        korean_text = text_item.get("text", "")
                    else:
                        korean_text = str(text_item)
                    
                    translated_text = translations.get(korean_text, "")
                    print(f"[ImageProcessor] Region {idx}: bbox=({x_min},{y_min},{w},{h}), '{korean_text}' -> '{translated_text}'")
                else:
                    print(f"[ImageProcessor] Region {idx}: No corresponding recognized text (idx {idx} >= {len(recognized_texts)})")
                
                if not translated_text:
                    print(f"[ImageProcessor] Region {idx}: Skipping - no translation found")
                    continue
                
                # Split text into lines
                words = translated_text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                    
                    if text_size[0] > w - 10:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                
                if current_line:
                    lines.append(current_line)
                
                # Draw lines centered
                line_height = TEXT_LINE_HEIGHT
                total_text_height = len(lines) * line_height
                y_offset = y_min + (h - total_text_height) // 2 + line_height // 2
                
                for line in lines:
                    if y_offset + line_height > y_max:
                        break
                    text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                    x_centered = x_min + (w - text_size[0]) // 2
                    
                    cv2.putText(
                        image,
                        line,
                        (x_centered, y_offset),
                        font,
                        font_scale,
                        font_color,
                        thickness,
                    )
                    y_offset += line_height
            
            return True
        except Exception as e:
            print(f"[ImageProcessor] Error drawing translations: {e}")
            return False
