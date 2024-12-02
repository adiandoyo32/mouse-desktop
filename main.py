import sys
import cv2
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from cvzone.HandTrackingModule import HandDetector
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from collections import deque
import time
import numpy as np
from keras.models import load_model
import threading

from ui import Ui_MainWindow

detector = HandDetector(detectionCon=0.8, maxHands=1)
mouse = Controller()

# Get screen size using screeninfo
monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height

frame_width = 640
frame_height = 480

prev_time = 0
frame_rate = 60  # Desired frame rate

# Speed factor to control the sensitivity of mouse movement
speed_factor = 5.0

# Moving average window size
smoothing_window = 10
x_deque = deque(maxlen=smoothing_window)
y_deque = deque(maxlen=smoothing_window)

# Load the model
model = load_model("model/keras_model.h5", compile=False)

# Load the labels
class_names = open("model/labels.txt", "r").readlines()
print(class_names)

# Variable to track time of last click
delay = 0

def leftClickDelay():
  global delay
  global clickThread
  time.sleep(1)
  delay = 0
  clickThread = threading.Thread(target=leftClickDelay)
  
clickThread  = threading.Thread(target=leftClickDelay)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.setTabIndexHome)
        self.pushButton_2.clicked.connect(self.setTabIndexDeskripsi)
        self.pushButton_3.clicked.connect(self.start)
        self.pushButton_4.clicked.connect(self.stop)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, frame_width)
        self.cap.set(4, frame_height)
        
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_frame)

    def setTabIndexHome(self):
        self.tabWidget.setCurrentIndex(0)

    def setTabIndexDeskripsi(self):
        self.tabWidget.setCurrentIndex(1)

    def start(self):
        self.timer.start(20)
        
    def update_frame(self):
        if self.cap.isOpened():
            start_time = time.time()
            
            # Read captured camera
            ret, frame = self.cap.read()
            
            # Flip the image horizontally
            frame = cv2.flip(frame, 1)  
            hands, img = detector.findHands(frame, draw=False)
            
            if hands:
                hand = hands[0]
                lmList = hand["lmList"]  # List of 21 Landmark points
                
                # Calculate the centroid of the hand
                x_coords = [lm[0] for lm in lmList]
                y_coords = [lm[1] for lm in lmList]
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))

                # Append to the deque for smoothing
                x_deque.append(centroid_x)
                y_deque.append(centroid_y)

                # Calculate the smoothed centroid
                smoothed_x = int(np.mean(x_deque))
                smoothed_y = int(np.mean(y_deque))

                # Draw the smoothed centroid
                cv2.circle(frame, (smoothed_x, smoothed_y), 10, (0, 255, 0), cv2.FILLED)
                
                # Apply adaptive scaling
                scaled_x = self.adaptive_scaling(smoothed_x, frame_width)
                scaled_y = self.adaptive_scaling(smoothed_y, frame_height)

                # Map the scaled coordinates to screen coordinates
                screen_x = np.interp(scaled_x, (0, frame_width), (0, screen_width * 1.5))
                screen_y = np.interp(scaled_y, (0, frame_height), (0, screen_height * 1.5))

                # Move the mouse
                mouse.position = (screen_x, screen_y)
                
                self.predict_gestures(frame)
            
            if ret:
                # Convert the frame to RGB format
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert the RGB image to QImage
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Convert QImage to QPixmap and set it on the label
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
              # Calculate and display FPS
            end_time = time.time()
            fps = int(1 / (end_time - start_time))
            cv2.putText(frame, f"FPS: {fps}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def stop(self):
        self.timer.stop()
        self.video_label.clear()
        
    def adaptive_scaling(self, value, max_value):
        """
        Apply an adaptive scaling to the input value to improve control near edges.
        """
        return (value / max_value) ** 2 * max_value

    def predict_gestures(self, frame):
        global delay
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        if (class_name[0] == '0'):
            print('index')
            # mouse.scroll(0, 10)
        elif class_name[0] == '1':
            if delay == 0:
                print('scrolll')
                mouse.scroll(0, 1)
                # mouse.click(Button.left, 1)
                delay = 1
                clickThread.start()
        elif (class_name[0] == '2'):
            print('kepal')
            # mouse.click(Button.right, 1)
            
        cv2.putText(frame, f"Gesture: {class_name[2:]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()