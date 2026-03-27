from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from gap_detection_gui import GapDetectionModule
import sys


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主控台")
        self.resize(640, 420)
        self.setMinimumSize(560, 360)

        central = QWidget(self)
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.addStretch(1)

        btn = QPushButton("打开林窗检测分析模块")
        btn.setMinimumSize(280, 56)
        btn.clicked.connect(self.open_gap_module)
        layout.addWidget(btn, alignment=Qt.AlignHCenter)

        layout.addStretch(1)

    def open_gap_module(self):
        # 实例化并使用 exec_() 作为模态窗口弹出
        self.gap_dialog = GapDetectionModule(self)
        self.gap_dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())