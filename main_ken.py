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
from keras.models import model_from_json
import threading

from ui import Ui_MainWindow

detector = HandDetector(detectionCon=0.8, maxHands=1)
mouse = Controller()

# Get screen size using screeninfo
monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height

frame_width = 640
frame_height = 480

# Moving average window size
smoothing_window = 10
x_deque = deque(maxlen=smoothing_window)
y_deque = deque(maxlen=smoothing_window)

# Load model architecture from JSON file
with open("model/model_1.json", "r") as json_file:
    model_json = json_file.read()

# Load model from JSON
model = model_from_json(model_json)

# Load weights into the model
model.load_weights("model/model_1.h5")

# Load the labels
label = ['fist','five','four','index', 'ok', 'v']
# labels = open("model/labels.txt", "r").readlines()

left_click_delay = 0
def delayed_left_click():
    global left_click_delay
    global left_click_thread
    time.sleep(1)
    left_click_delay = 0
    left_click_thread = threading.Thread(target=delayed_left_click)
left_click_thread  = threading.Thread(target=delayed_left_click)

right_click_delay = 0
def delayed_right_click():
    global right_click_delay
    global right_click_thread
    time.sleep(1)
    right_click_delay = 0
    right_click_thread = threading.Thread(target=delayed_right_click)
right_click_thread  = threading.Thread(target=delayed_right_click)

double_click_delay = 0
def delayed_double_click():
    global double_click_delay
    global double_click_thread
    time.sleep(1)
    double_click_delay = 0
    double_click_thread = threading.Thread(target=delayed_double_click)
double_click_thread  = threading.Thread(target=delayed_double_click)

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
        self.timer.timeout.connect(self.update_frame)

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
                self.move_mouse(hand, img)
                
                # extract hand roi
                hand_roi = self.process_hand_roi(hand, img)
                
                # predict gesture
                self.predict_gesture(hand_roi, img)

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
            self.calculate_fps(start_time, end_time, img)
            
    def move_mouse(self, hand, frame):
        # List of 21 Landmark points
        lmList = hand["lmList"]
        
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
        screen_x = np.interp(scaled_x, (0, frame_width), (0, screen_width))
        screen_y = np.interp(scaled_y, (0, frame_height), (0, screen_height))

        # Move the mouse
        mouse.position = (screen_x, screen_y)
        
    def process_hand_roi(self, hand, frame):
        copy_frame = frame.copy()
        bbox = hand['bbox']
        
        # Bounding box: x, y, w, h
        x, y, w, h = bbox

        yCoord = y - 20 if y - 20 > 0 else 0
        xCoord = x - 20 if x - 20 > 0 else 0
        heightRoi = y + h + 20
        widthRoi = x + w + 20
        
        cropframe = copy_frame[yCoord : heightRoi, xCoord : widthRoi]    
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe,(48, 48))
        
        # Extract feature
        cropframe = self.extract_features(cropframe)
        
        cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
        
        return cropframe

    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1,48,48,1)
        return feature/255.0

    def adaptive_scaling(self, value, max_value):
        # Apply an adaptive scaling to the input value to improve control near edges.
        return (value / max_value) ** 2 * max_value * 3

    def predict_gesture(self, hand_roi, frame):
        pred = model.predict(hand_roi)
        print(pred.argmax())
        prediction_label = label[pred.argmax()]
        
        # Check prediction accuracy, if under 90 then return
        accuracy = np.max(pred)*100
        cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
        cv2.putText(frame, f'{prediction_label}  {accuracy:.2f}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
        if (accuracy < 90):
            return
        
        
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 60),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
        else:
            self.gesture_action(prediction_label)
            
            
    def gesture_action(self, label):
        global left_click_delay, right_click_delay, double_click_delay
        if (label == 'index'):
            if (left_click_delay == 0):
                left_click_delay = 1
                left_click_thread.start()
                print('index left click')
                # mouse.click(Button.left, 1)
        elif (label == 'v'):
            if (right_click_delay == 0):
                right_click_delay = 1
                right_click_thread.start()
                print('v right click')
                # mouse.click(Button.right, 1)
        elif (label == 'ok'):
            if (double_click_delay == 0):
                double_click_delay = 1
                double_click_thread.start()
                print('ok double click')
                # mouse.click(Button.left, 2)
        elif (label == 'four'):
            print('four scroll up')
            # mouse.scroll(0, -1)
        elif (label == 'five'):
            print('five scroll down')
            # mouse.scroll(0, -1)
        elif (label == 'fist'):
            print('fist')
            
    def calculate_fps(self, start_time, end_time, frame):
        fps = int(1 / (end_time - start_time))
        cv2.putText(frame, f"FPS: {fps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    def stop(self):
        self.timer.stop()
        self.video_label.clear()


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()