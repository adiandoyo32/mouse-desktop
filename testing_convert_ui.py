import sys
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from testing_convert_to_exe import Ui_Form


class Login(QWidget):
    def __init__(self):
        super().__init__()

        # use the Ui_login_form
        self.ui = Ui_Form()       
        self.ui.setupUi(self)       
        
        self.ui.pushButton.clicked.connect(self.clickButton)
        
        # show the login window
        self.show()
        
    def clickButton(self):
        text = self.ui.lineEdit.text()
        QMessageBox.information(self, 'Success', text)
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = Login()
    sys.exit(app.exec())