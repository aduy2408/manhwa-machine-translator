"""Entry point for the Korean OCR & Translation application."""

import sys
from dotenv import load_dotenv
from PyQt5.QtWidgets import QApplication
from gui import MainWindow


def main():
    """Start the application."""
    # Load environment variables from .env file
    load_dotenv()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
