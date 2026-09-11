"""Qt application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gst_image_app.mainwindow import MainWindow


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("GST Image")
    application.setOrganizationName("Group Six Technologies")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
