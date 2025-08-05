import sys
sys.path.append('./')
sys.path.append('../')
from src.main_window import MainWindow
from PyQt5 import QtWidgets


def main():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    main()