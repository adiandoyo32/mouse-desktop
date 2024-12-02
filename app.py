import sys 
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
from ui import Ui_mainForm
import cv2

class Webcam:
    def __init__(self):
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_faces)
        
    def start_webcam(self):
        self.video_capture = cv2.VideoCapture(0)  # 0 corresponds to the default webcam
        self.timer.start(100)  # Set the timer to trigger every 100 ms (adjust as needed)
    
    def stop_webcam(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.timer.stop()
            
    def detect(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                self.ui.setPixmap(pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
          

class App(QWidget):
    def __init__(self):
        super().__init__()

        # use the Ui_login_form
        self.ui = Ui_mainForm()       
        self.ui.setupUi(self)
        
        self.frameTimer = QTimer()
        self.frameTimer.timeout.connect(self.detect)
        
        # connect button
        self.ui.clickButton.clicked.connect(self.startWebcam)
        
        # show the login window
        self.show()
      
    def print(self):
        text = self.ui.input.text()
        print(text)
        
    def startWebcam(self):
        self.frameTimer.start(100)
        
    def detect(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        coba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(coba, coba.shape[1], coba.shape[0], QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label.setPixmap(pixmap)
        
  
if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = App()
    sys.exit(app.exec())