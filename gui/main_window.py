"""Main application window."""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from translator import Translator
from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    LAST_OPENED_PATH_FILE,
    get_last_opened_path,
    save_last_opened_path,
)

from .canvas import ImageCanvas, ScrollCanvas
from .workers import AutoTranslateWorker
from .scroll_manager import ScrollModeManager
from .text_detector import TextDetector


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Korean OCR & Translation Tool")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Initialize components
        try:
            self.translator = Translator()
        except ValueError as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)

        self.canvas = ImageCanvas()
        self.ocr_worker = None
        self.translation_worker = None
        self.auto_translate_worker = None
        self.is_scroll_mode = False
        self.scroll_mode_manager = None
        self.current_ocr_canvas = None
        self.text_detector = TextDetector()
        self.detected_regions = {}  # image_index -> bboxes
        self.current_auto_translate_canvas = None
        self.current_auto_translate_image_idx = None

        # Setup UI
        self.init_ui()
        self.setup_shortcuts()

    def init_ui(self):
        """Initialize the user interface."""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QHBoxLayout()

        # Left side - Image canvas with scroll area
        left_layout = QVBoxLayout()

        # Add canvas to scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setStyleSheet("border: 1px solid black;")
        left_layout.addWidget(self.scroll_area)

        # Image control buttons
        img_button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        self.save_btn = QPushButton("💾 Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.detect_text_btn = QPushButton("🔍 Translate Current")
        self.detect_text_btn.clicked.connect(self.auto_translate_detected_text)
        self.detect_text_btn.setEnabled(False)
        self.translate_all_btn = QPushButton("🔍 Translate All")
        self.translate_all_btn.clicked.connect(self.auto_translate_all_pages)
        self.translate_all_btn.setEnabled(False)
        self.scroll_mode_btn = QPushButton("📖 Scroll Mode OFF")
        self.scroll_mode_btn.clicked.connect(self.toggle_scroll_mode)
        self.scroll_mode_btn.setEnabled(False)
        img_button_layout.addWidget(self.load_btn)
        img_button_layout.addWidget(self.load_folder_btn)
        img_button_layout.addWidget(self.save_btn)
        img_button_layout.addWidget(self.detect_text_btn)
        img_button_layout.addWidget(self.translate_all_btn)
        img_button_layout.addWidget(self.scroll_mode_btn)
        left_layout.addLayout(img_button_layout)

        # Navigation buttons for folder images (wrap in widget so we can hide it)
        self.nav_widget = QWidget()
        self.nav_button_layout = QHBoxLayout()
        self.prev_img_btn = QPushButton("← Previous")
        self.prev_img_btn.clicked.connect(self.prev_image)
        self.prev_img_btn.setEnabled(False)
        self.image_info_label = QLabel("No images loaded")
        self.next_img_btn = QPushButton("Next →")
        self.next_img_btn.clicked.connect(self.next_image)
        self.next_img_btn.setEnabled(False)
        self.nav_button_layout.addWidget(self.prev_img_btn)
        self.nav_button_layout.addWidget(self.image_info_label)
        self.nav_button_layout.addWidget(self.next_img_btn)
        self.nav_widget.setLayout(self.nav_button_layout)
        left_layout.addWidget(self.nav_widget)

        # Right side - Info (compact)
        right_layout = QVBoxLayout()

        # Info label
        right_layout.addWidget(QLabel("Auto-Translation"))
        info_label = QLabel(
            "Click 'Auto Translate' to automatically detect Korean text "
            "and replace it with English translations.\n\n"
            "Use Scroll Mode for processing multiple images."
        )
        info_label.setWordWrap(True)
        right_layout.addWidget(info_label)

        # Add stretch to push everything to top
        right_layout.addStretch()

        # Add to main layout (image takes 4x more space)
        main_layout.addLayout(left_layout, 4)
        main_layout.addLayout(right_layout, 1)

        main_widget.setLayout(main_layout)

    def load_image(self):
        """Load image from file."""
        last_path = get_last_opened_path()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", last_path, "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path:
            try:
                # Save directory of opened file
                save_last_opened_path(os.path.dirname(file_path))

                self.canvas.load_image(file_path)
                self.canvas.update_display()
                self.detect_text_btn.setEnabled(True)
                self.translate_all_btn.setEnabled(False)  # Only for folders
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def load_folder(self):
        """Load folder of images and display them one by one."""
        last_path = get_last_opened_path()
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Folder with Images", last_path
        )

        if folder_path:
            try:
                # Save opened folder
                save_last_opened_path(folder_path)

                self.canvas.image_processor.load_folder(folder_path)
                self.canvas.update_display()
                self.detect_text_btn.setEnabled(True)
                self.translate_all_btn.setEnabled(True)

                # Update navigation buttons
                self.update_nav_buttons()

                num_images = len(self.canvas.image_processor.image_list)
                # Enable scroll mode and detect text buttons when folder is loaded
                self.scroll_mode_btn.setEnabled(True)
                QMessageBox.information(self, "Success", f"Loaded {num_images} images from folder")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load folder: {str(e)}")

    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Ctrl+Z to undo/clear drawing
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.canvas.reset_drawing)

        # Arrow keys for navigation
        left_shortcut = QShortcut(QKeySequence("Left"), self)
        left_shortcut.activated.connect(self.prev_image)
        right_shortcut = QShortcut(QKeySequence("Right"), self)
        right_shortcut.activated.connect(self.next_image)

        # Page Up/Down for faster navigation
        page_up_shortcut = QShortcut(QKeySequence("Page Up"), self)
        page_up_shortcut.activated.connect(self.prev_image)
        page_down_shortcut = QShortcut(QKeySequence("Page Down"), self)
        page_down_shortcut.activated.connect(self.next_image)

        # Home/End for first/last image
        home_shortcut = QShortcut(QKeySequence("Home"), self)
        home_shortcut.activated.connect(self.first_image)
        end_shortcut = QShortcut(QKeySequence("End"), self)
        end_shortcut.activated.connect(self.last_image)

    def update_nav_buttons(self):
        """Update navigation button states."""
        has_images = len(self.canvas.image_processor.image_list) > 0
        self.prev_img_btn.setEnabled(has_images and self.canvas.image_processor.current_image_index > 0)
        self.next_img_btn.setEnabled(
            has_images
            and self.canvas.image_processor.current_image_index
            < len(self.canvas.image_processor.image_list) - 1
        )
        self.image_info_label.setText(self.canvas.image_processor.get_image_info())

    def prev_image(self):
        """Go to previous image."""
        if self.canvas.image_processor.prev_image():
            self.canvas.update_display()
            self.update_nav_buttons()
            self.reset_scroll()

    def next_image(self):
        """Go to next image."""
        if self.canvas.image_processor.next_image():
            self.canvas.update_display()
            self.update_nav_buttons()
            self.reset_scroll()

    def first_image(self):
        """Go to first image."""
        if self.canvas.image_processor.image_list:
            self.canvas.image_processor.current_image_index = 0
            self.canvas.image_processor.original_image = self.canvas.image_processor.image_list[0]
            self.canvas.image_processor.display_image = self.canvas.image_processor.original_image.copy()
            self.canvas.image_processor.mask = np.zeros(
                self.canvas.image_processor.display_image.shape[:2], dtype=np.uint8
            )
            self.canvas.update_display()
            self.update_nav_buttons()
            self.reset_scroll()
            print("[GUI] Jumped to first image")

    def last_image(self):
        """Go to last image."""
        if self.canvas.image_processor.image_list:
            last_idx = len(self.canvas.image_processor.image_list) - 1
            self.canvas.image_processor.current_image_index = last_idx
            self.canvas.image_processor.original_image = self.canvas.image_processor.image_list[last_idx]
            self.canvas.image_processor.display_image = self.canvas.image_processor.original_image.copy()
            self.canvas.image_processor.mask = np.zeros(
                self.canvas.image_processor.display_image.shape[:2], dtype=np.uint8
            )
            self.canvas.update_display()
            self.update_nav_buttons()
            self.reset_scroll()
            print(f"[GUI] Jumped to last image ({last_idx + 1})")

    def reset_scroll(self):
        """Reset scroll area to top."""
        self.scroll_area.verticalScrollBar().setValue(0)

    def toggle_scroll_mode(self):
        """Toggle between single image and continuous scroll mode."""
        if not self.canvas.image_processor.image_list:
            return

        self.is_scroll_mode = not self.is_scroll_mode

        if self.is_scroll_mode:
            self._setup_scroll_mode()
            self.scroll_mode_btn.setText("📖 Scroll Mode ON")
            self.nav_widget.setVisible(False)
        else:
            self._teardown_scroll_mode()
            self.scroll_mode_btn.setText("📖 Scroll Mode OFF")
            self.nav_widget.setVisible(True)
            self.canvas.update_display()

    def _setup_scroll_mode(self):
        """Setup scroll mode manager."""
        self.scroll_mode_manager = ScrollModeManager(
            self.scroll_area,
            self.canvas.image_processor.image_list,
            self._on_scroll_canvas_active,
        )
        self.scroll_mode_manager.setup()

    def _teardown_scroll_mode(self):
        """Teardown scroll mode."""
        if self.scroll_mode_manager:
            self.scroll_mode_manager.teardown()
            self.scroll_area.setWidget(self.canvas)
            self.reset_scroll()
            self.scroll_mode_manager = None

    def _on_scroll_canvas_active(self, canvas_idx):
        """Called when a canvas in scroll mode becomes active."""
        if self.scroll_mode_manager:
            self.scroll_mode_manager.set_active_canvas(canvas_idx)
            print(f"[GUI] Active canvas: {canvas_idx + 1}")



    def auto_translate_detected_text(self):
        """Auto-detect, translate, and draw text on the current image."""
        if not self.canvas.image_processor.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        # Get current image
        if self.is_scroll_mode and self.scroll_mode_manager:
            current_idx = self.scroll_mode_manager.active_canvas_idx
            current_image = self.canvas.image_processor.image_list[current_idx]
        else:
            current_idx = 0
            current_image = self.canvas.image_processor.original_image

        self.detect_text_btn.setEnabled(False)
        self.detect_text_btn.setText("🔍 Translating...")

        # Store for later use in callback
        self.current_auto_translate_image_idx = current_idx

        # Start worker
        self.auto_translate_worker = AutoTranslateWorker(current_image, self.text_detector, self.translator)
        self.auto_translate_worker.finished.connect(self.on_auto_translate_finished)
        self.auto_translate_worker.error.connect(self.on_auto_translate_error)
        self.auto_translate_worker.start()

    def auto_translate_all_pages(self):
        """Auto-detect and translate all pages with context for consistency."""
        if not self.canvas.image_processor.image_list or len(self.canvas.image_processor.image_list) == 0:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        self.detect_text_btn.setEnabled(False)
        self.detect_text_btn.setText("🔍 Translating...")
        self.translate_all_btn.setEnabled(False)
        self.translate_all_btn.setText("🔍 Translating...")

        # Store all images and their detected texts
        self.all_pages_data = []  # Will store (image_idx, image, polygons, recognized_texts) for each page

        # Detect text in all images first
        print(f"[AutoTranslateAll] Detecting text in {len(self.canvas.image_processor.image_list)} images...")
        for idx, image in enumerate(self.canvas.image_processor.image_list):
            result = self.text_detector.detect_and_recognize_text(image)
            if result:
                polygons, recognized_texts = result
                self.all_pages_data.append((idx, image, polygons, recognized_texts))
                print(f"[AutoTranslateAll] Image {idx + 1}: Detected {len(recognized_texts)} texts")
            else:
                print(f"[AutoTranslateAll] Image {idx + 1}: No text detected")

        if not self.all_pages_data:
            QMessageBox.information(self, "Info", "No text detected in any image.")
            self.detect_text_btn.setEnabled(True)
            self.detect_text_btn.setText("🔍 Translate Current")
            self.translate_all_btn.setEnabled(True)
            self.translate_all_btn.setText("🔍 Translate All")
            return

        # Extract all texts from all pages for batch translation
        all_korean_texts = []
        for _, _, _, recognized_texts in self.all_pages_data:
            for item in recognized_texts:
                korean_text = item["text"]
                if korean_text not in all_korean_texts:  # Avoid duplicates
                    all_korean_texts.append(korean_text)

        print(f"[AutoTranslateAll] Total unique texts to translate: {len(all_korean_texts)}")

        # Batch translate all texts at once for consistency
        translations = self.translator.translate_batch(all_korean_texts)
        print(f"[AutoTranslateAll] Received {len(translations)} translations")

        if not translations:
            QMessageBox.warning(self, "Warning", "Translation failed.")
            self.detect_text_btn.setEnabled(True)
            self.detect_text_btn.setText("🔍 Translate Current")
            self.translate_all_btn.setEnabled(True)
            self.translate_all_btn.setText("🔍 Translate All")
            return

        # Draw translations on all images
        from image_processor import ImageProcessor
        for image_idx, image, polygons, recognized_texts in self.all_pages_data:
            image_with_translations = self.canvas.image_processor.image_list[image_idx].copy()
            ImageProcessor.draw_translations_on_image(
                image_with_translations, polygons, recognized_texts, translations
            )
            self.canvas.image_processor.image_list[image_idx] = image_with_translations

        # Update display
        if self.is_scroll_mode and self.scroll_mode_manager:
            # Update all scroll canvases
            for idx, canvas in enumerate(self.scroll_mode_manager.scroll_canvases):
                canvas.original_image = self.canvas.image_processor.image_list[idx]
                canvas.display_image = self.canvas.image_processor.image_list[idx].copy()
                canvas.mask = np.zeros(self.canvas.image_processor.image_list[idx].shape[:2], dtype=np.uint8)
                canvas.update_display()
        else:
            # Update single canvas
            self.canvas.image_processor.original_image = self.canvas.image_processor.image_list[0]
            self.canvas.image_processor.display_image = self.canvas.image_processor.image_list[0].copy()
            self.canvas.update_display()

        QMessageBox.information(
            self,
            "Success",
            f"Auto-translated all {len(self.all_pages_data)} pages with consistent translations."
        )

        self.detect_text_btn.setEnabled(True)
        self.detect_text_btn.setText("🔍 Translate Current")
        self.translate_all_btn.setEnabled(True)
        self.translate_all_btn.setText("🔍 Translate All")

    def on_auto_translate_finished(self, polygons, recognized_texts, translations):
        """Handle auto-translate completion."""
        try:
            current_idx = self.current_auto_translate_image_idx

            # Step 4: Draw translations on image
            from image_processor import ImageProcessor
            target_image = self.canvas.image_processor.image_list[current_idx]
            image_with_translations = target_image.copy()
            ImageProcessor.draw_translations_on_image(image_with_translations, polygons, recognized_texts, translations)
            
            # Update the image in the list with the translated version
            self.canvas.image_processor.image_list[current_idx] = image_with_translations
            self.canvas.image_processor.original_image = image_with_translations
            self.canvas.image_processor.display_image = image_with_translations.copy()

            # Update display
            if self.is_scroll_mode and self.scroll_mode_manager:
                canvas = self.scroll_mode_manager.scroll_canvases[current_idx]
                canvas.original_image = image_with_translations
                canvas.display_image = image_with_translations.copy()
                canvas.mask = np.zeros(image_with_translations.shape[:2], dtype=np.uint8)
                canvas.update_display()
            else:
                self.canvas.update_display()

            korean_texts = [item["text"] for item in recognized_texts]
            QMessageBox.information(
                self,
                "Success",
                f"Auto-translated {len(korean_texts)} text regions."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to draw translations: {str(e)}")
        finally:
            self.detect_text_btn.setEnabled(True)
            self.detect_text_btn.setText("🔍 Auto Translate")

    def on_auto_translate_error(self, error_msg: str):
        """Handle auto-translate error."""
        QMessageBox.critical(self, "Auto-Translate Error", f"Error: {error_msg}")
        self.detect_text_btn.setEnabled(True)
        self.detect_text_btn.setText("🔍 Auto Translate")



    def save_image(self):
        """Save the current image (with translated text) to a folder."""
        # Get the image to save
        if self.is_scroll_mode:
            canvas = self._get_active_canvas()
            if not isinstance(canvas, ScrollCanvas):
                QMessageBox.warning(self, "Warning", "No canvas available.")
                return
            image_to_save = canvas.original_image
            img_num = canvas.index + 1
        else:
            image_to_save = self.canvas.image_processor.original_image
            img_num = 1

        if image_to_save is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        # Ask user where to save
        folder_path = QFileDialog.getExistingDirectory(
            self, "Save image to folder"
        )

        if folder_path:
            try:
                import cv2
                filename = f"translated_image_{img_num}.png"
                filepath = os.path.join(folder_path, filename)
                cv2.imwrite(filepath, image_to_save)
                QMessageBox.information(
                    self, "Success", f"Image saved to:\n{filepath}"
                )
                print(f"[save_image] Saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def clear_cache(self):
        """Clear persistent cache (last opened path)."""
        try:
            if os.path.exists(LAST_OPENED_PATH_FILE):
                os.remove(LAST_OPENED_PATH_FILE)
                QMessageBox.information(self, "Success", "Cache cleared successfully")
                print("[GUI] Cache cleared")
            else:
                QMessageBox.information(self, "Info", "No cache to clear")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear cache: {str(e)}")



    def _get_active_canvas(self):
        """Get the currently active canvas (scroll or regular)."""
        if self.is_scroll_mode and self.scroll_mode_manager:
            return self.scroll_mode_manager.get_active_canvas()
        return self.canvas
