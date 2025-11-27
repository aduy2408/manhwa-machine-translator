"""Canvas widgets for image display and brush drawing."""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from image_processor import ImageProcessor
from config import BRUSH_SIZE, BRUSH_COLOR, RECTANGLE_PADDING


class BaseCanvas(QLabel):
    """Base class for canvas widgets with brush drawing support."""

    def __init__(self, brush_color=BRUSH_COLOR):
        super().__init__()
        self.brush_size = BRUSH_SIZE
        self.brush_color = brush_color
        self.drawing = False
        self.drawing_enabled = True  # Toggle for drawing mode
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)

    def update_display(self):
        """Override in subclasses to update the displayed image."""
        raise NotImplementedError

    def draw_at(self, pos):
        """Override in subclasses to handle brush drawing."""
        raise NotImplementedError

    def get_selected_region(self):
        """Override in subclasses to extract selected region."""
        raise NotImplementedError

    def reset_drawing(self):
        """Override in subclasses to clear the drawing."""
        raise NotImplementedError

    def set_drawing_enabled(self, enabled: bool):
        """Enable or disable drawing mode."""
        self.drawing_enabled = enabled
        if not enabled:
            self.drawing = False

    def mousePressEvent(self, event):
        """Drawing disabled."""
        pass

    def mouseMoveEvent(self, event):
        """Drawing disabled."""
        pass

    def mouseReleaseEvent(self, event):
        """Drawing disabled."""
        pass


class ImageCanvas(BaseCanvas):
    """Canvas for displaying a single image with brush input."""

    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()
        self.setStyleSheet("border: 1px solid black;")

    def load_image(self, image_path: str):
        """Load and display an image."""
        self.image_processor.load_image(image_path)
        self.update_display()

    def update_display(self):
        """Update the displayed image."""
        image = self.image_processor.get_display_image()
        self._render_image(image)

    def draw_at(self, pos):
        """Draw brush stroke at given position."""
        if self.image_processor.original_image is None or self.pixmap() is None:
            return

        x = pos.x()
        y = pos.y()
        pixmap = self.pixmap()

        if pixmap.width() <= 0 or pixmap.height() <= 0:
            return

        if x < 0 or x >= pixmap.width() or y < 0 or y >= pixmap.height():
            return

        img_x = max(0, min(int(x), self.image_processor.display_image.shape[1] - 1))
        img_y = max(0, min(int(y), self.image_processor.display_image.shape[0] - 1))

        self.image_processor.draw_brush_stroke(img_x, img_y, self.brush_size, self.brush_color)
        self.update_display()

    def get_selected_region(self):
        """Extract selected region."""
        return self.image_processor.get_selected_region()

    def reset_drawing(self):
        """Reset the drawing (but preserve bbox for text replacement)."""
        print("[ImageCanvas.reset_drawing] Clearing mask")
        self.image_processor.reset_mask()
        # Note: bbox is preserved in image_processor.last_selection_bbox for translation
        self.update_display()
        print("[ImageCanvas.reset_drawing] Done")

    def _render_image(self, image):
        """Convert numpy image to QPixmap and display."""
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = 3 * w
            q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.setPixmap(pixmap)
            self.adjustSize()


class ScrollCanvas(BaseCanvas):
    """Canvas for a single image in scroll mode (multi-image view)."""

    def __init__(self, image, index, on_active_callback):
        super().__init__()
        self.original_image = image  # Reference to original image in list (kept pristine)
        self.display_image = image.copy()  # Visual copy for brush strokes
        self.index = index
        self.on_active_callback = on_active_callback
        self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
        self.last_selection_bbox = None
        self.detected_regions = []  # Store detected text regions as bboxes
        self.setStyleSheet("border: 1px solid gray;")
        self.update_display()

    def update_display(self):
        """Update the displayed image."""
        self._render_image(self.display_image)

    def draw_at(self, pos):
        """Draw brush stroke at given position."""
        if self.display_image is None or self.pixmap() is None:
            return

        self.on_active_callback(self.index)

        x = pos.x()
        y = pos.y()
        pixmap = self.pixmap()

        if pixmap.width() <= 0 or pixmap.height() <= 0:
            return

        if x < 0 or x >= pixmap.width() or y < 0 or y >= pixmap.height():
            return

        img_x = max(0, min(int(x), self.display_image.shape[1] - 1))
        img_y = max(0, min(int(y), self.display_image.shape[0] - 1))

        # Draw on mask and display image only (keep original_image pristine for OCR)
        cv2.circle(self.mask, (img_x, img_y), self.brush_size, 255, -1)
        cv2.circle(self.display_image, (img_x, img_y), self.brush_size, self.brush_color, -1)

        self.update_display()

    def get_selected_region(self):
        """Extract selected region from original image (without brush strokes)."""
        if self.mask is None or self.mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        x = max(0, x - RECTANGLE_PADDING)
        y = max(0, y - RECTANGLE_PADDING)
        w = min(self.original_image.shape[1] - x, w + 2 * RECTANGLE_PADDING)
        h = min(self.original_image.shape[0] - y, h + 2 * RECTANGLE_PADDING)

        self.last_selection_bbox = (x, y, w, h)
        # Extract from original_image (pristine, without brush strokes)
        region = self.original_image[y : y + h, x : x + w]
        return region

    def reset_drawing(self):
        """Clear the drawing (but preserve bbox for text replacement)."""
        print(f"[ScrollCanvas.reset_drawing] Clearing canvas {self.index}")
        # Restore display_image from pristine original_image
        self.display_image = self.original_image.copy()
        self.mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
        # Keep last_selection_bbox for translation - don't clear it
        self.update_display()
        print(f"[ScrollCanvas.reset_drawing] Done for canvas {self.index}")

    def highlight_detected_regions(self, bboxes: list):
        """Draw rectangles around detected text regions."""
        if not bboxes or self.display_image is None:
            return

        self.detected_regions = bboxes
        for bbox in bboxes:
            x, y, w, h = bbox
            # Draw rectangle on display image
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.update_display()
        print(f"[ScrollCanvas.highlight_detected_regions] Drew {len(bboxes)} regions on canvas {self.index}")

    def replace_region_with_text(self, translated_text: str) -> bool:
        """Replace selected region with white box and translated text."""
        print(f"[ScrollCanvas.replace_region_with_text] Canvas {self.index}, bbox: {self.last_selection_bbox}")
        if self.last_selection_bbox is None:
            print(f"[ScrollCanvas.replace_region_with_text] No bbox found!")
            return False
        result = ImageProcessor.draw_text_on_images(
            self.original_image, self.display_image, self.last_selection_bbox, translated_text
        )
        print(f"[ScrollCanvas.replace_region_with_text] draw_text_on_images returned: {result}")
        # Refresh display to show the translated text
        self.update_display()
        print(f"[ScrollCanvas.replace_region_with_text] update_display called")
        return result

    def _render_image(self, image):
        """Convert numpy image to QPixmap and display."""
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = 3 * w
            q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.setPixmap(pixmap)
