"""GUI module for Korean OCR & Translation Tool.

Architecture:
- canvas.py: Canvas widgets (BaseCanvas, ImageCanvas, ScrollCanvas)
- workers.py: Threading workers (OCRWorker, TranslationWorker)
- scroll_manager.py: ScrollModeManager for multi-image view
- main_window.py: MainWindow orchestrating the application
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]
