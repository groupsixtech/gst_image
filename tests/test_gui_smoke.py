from gst_image_app.mainwindow import MainWindow


def test_main_window_constructs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert "GST Image" in window.windowTitle()
    assert window.preview_button.text() == "Preview"
    assert window.run_button.text() == "Run particles"
