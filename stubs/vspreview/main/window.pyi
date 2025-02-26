from PyQt6.QtWidgets import QMainWindow
from vstools import SPath

__all__ = ['MainWindow']


class MainWindow(QMainWindow):
    script_path: SPath
