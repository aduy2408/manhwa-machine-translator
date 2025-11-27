"""Scroll mode manager for multi-image view."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .canvas import ScrollCanvas


class ScrollModeManager:
    """Manages scroll mode setup and teardown for multi-image view."""

    def __init__(self, scroll_area, image_list, on_canvas_active):
        self.scroll_area = scroll_area
        self.image_list = image_list
        self.on_canvas_active = on_canvas_active
        self.scroll_canvases = []
        self.active_canvas_idx = 0

    def setup(self):
        """Create scrollable view with all images."""
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setSpacing(5)
        container_layout.setContentsMargins(5, 5, 5, 5)

        self.scroll_canvases = []
        for idx, img in enumerate(self.image_list):
            canvas = ScrollCanvas(img, idx, self.on_canvas_active)
            self.scroll_canvases.append(canvas)
            container_layout.addWidget(canvas)

        container_layout.addStretch()
        container.setLayout(container_layout)

        self.scroll_area.takeWidget()
        self.scroll_area.setWidget(container)
        self.active_canvas_idx = 0
        self.on_canvas_active(0)

        print("[ScrollModeManager] Scroll mode activated")

    def teardown(self):
        """Return to single image mode."""
        self.scroll_area.takeWidget()
        self.scroll_canvases = []
        print("[ScrollModeManager] Scroll mode deactivated")

    def get_active_canvas(self):
        """Get the currently active canvas."""
        if self.scroll_canvases and 0 <= self.active_canvas_idx < len(self.scroll_canvases):
            return self.scroll_canvases[self.active_canvas_idx]
        return None

    def set_active_canvas(self, idx):
        """Set the active canvas."""
        self.active_canvas_idx = idx
