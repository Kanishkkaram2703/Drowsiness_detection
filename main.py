import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

def start_alarm(sound):
    """Play the alarm sound"""
    playsound(sound)

class DrowsinessDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detector")
        
        self.alarm_on = False
        self.count = 0
        self.status1 = ''
        self.status2 = ''
        
        # Load model and cascades
        self.model = load_model(r"F:\\KANISHK\\projects_null_class\\drowsiness\\drowiness_.h5")
        self.face_cascade = cv2.CascadeClassifier(r"F:\\KANISHK\\projects_null_class\\drowsiness\\archive(1)\\haarcascade_frontalface_default.xml")
        self.left_eye_cascade = cv2.CascadeClassifier(r"F:\\KANISHK\\projects_null_class\\drowsiness\\archive(1)\\haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier(r"F:\\KANISHK\\projects_null_class\\drowsiness\\archive(1)\\haarcascade_righteye_2splits.xml")
        
        self.alarm_sound = r"F:\\KANISHK\\projects_null_class\\drowsiness\\archive(1)\\alarm.mp3"
        self.cap = cv2.VideoCapture(0)

        # GUI Elements
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.status_label = tk.Label(root, text="Status: Not Started", font=("Arial", 12))
        self.status_label.pack()

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection, width=20, font=("Arial", 10))
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED, width=20, font=("Arial", 10))
        self.stop_button.pack()

        # Set flags
        self.detecting = False

    def start_detection(self):
        self.detecting = True
        self.status_label.config(text="Status: Detecting...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.show_frame()

    def stop_detection(self):
        self.detecting = False
        self.status_label.config(text="Status: Stopped")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def preprocess_eye(self, eye_region):
        """Preprocess the eye region for model prediction."""
        eye = cv2.resize(eye_region, (145, 145))
        
        # Ensure eye has 3 channels by converting to RGB only if it is grayscale
        if len(eye.shape) == 2 or eye.shape[2] == 1:  # Check if it's grayscale
            eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        
        eye = eye.astype('float32') / 255.0  # Normalize
        eye = img_to_array(eye)
        eye = np.expand_dims(eye, axis=0)
        return eye

    def show_frame(self):
        if not self.detecting:
            return
        
        _, frame = self.cap.read()
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            left_eye = self.left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = self.right_eye_cascade.detectMultiScale(roi_gray)

            for (x1, y1, w1, h1) in left_eye:
                eye1 = roi_color[y1:y1 + h1, x1:x1 + w1]
                eye1 = self.preprocess_eye(eye1)  # Preprocess left eye
                pred1 = self.model.predict(eye1)
                self.status1 = np.argmax(pred1)
                break

            for (x2, y2, w2, h2) in right_eye:
                eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                eye2 = self.preprocess_eye(eye2)  # Preprocess right eye
                pred2 = self.model.predict(eye2)
                self.status2 = np.argmax(pred2)
                break

            # Condition for detecting closed eyes based on predictions
            if self.status1 == 0 and self.status2 == 0:  # Assuming 0 indicates 'Closed'
                self.count += 1
                if self.count >= 5:
                    self.status_label.config(text="Status: Drowsiness Alert!")
                    if not self.alarm_on:
                        self.alarm_on = True
                        t = Thread(target=start_alarm, args=(self.alarm_sound,))
                        t.daemon = True
                        t.start()
            else:
                self.status_label.config(text="Status: Eyes Open")
                self.count = 0
                self.alarm_on = False

        # Convert the frame to Tkinter format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Call this function again to keep the video feed live
        self.video_label.after(10, self.show_frame)

    def on_closing(self):
        self.detecting = False
        self.cap.release()
        self.root.destroy()

# Main tkinter setup
root = tk.Tk()
app = DrowsinessDetectorApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
