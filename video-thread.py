from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt6.QtGui import QPixmap
import sys
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
import mouse

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.delay = 0
        self.cam_w = 640
        self.cam_h = 480
        
    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:

                frame = cv2.flip(frame, 1)
                hands, frame = self.detector.findHands(frame, flipType=False)
                
                if hands:
                  lmList = hands[0]['lmList']
                  index_x, index_y = lmList[8][0], lmList[8][1]
                  
                  fingers = self.detector.fingersUp(hands[0])
                  # print(fingers)
                  
                  if fingers[0] == 1:
                    conv_x = int(np.interp(index_x, (0, self.cam_w), (0, 1536)))
                    conv_y = int(np.interp(index_y, (0, self.cam_h), (0, 864)))
                    mouse.move(conv_x, conv_y)
                    
                  # if fingers[1] == 1:
                  #   if delay == 0:
                  #     print('left click')
                  #     mouse.click(button='left')
                  #     delay = 1
                  #     clickThread.start()
                
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # face_cascade = cv2.CascadeClassifier(self.resource_path('haarcascade_frontalface_alt.xml'))
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
                # for (x, y, w, h) in faces:
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.change_pixmap_signal.emit(frame)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())