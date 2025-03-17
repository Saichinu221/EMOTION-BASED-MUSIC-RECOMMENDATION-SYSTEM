from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import os
from playsound import playsound
from PIL import Image, ImageTk
import threading

main = Tk()
main.title("EMOTION BASED MUSIC RECOMMENDATION SYSTEM")
main.geometry("1200x1200")

global filename
global faces
global frame
global img_label
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def upload():
    global filename
    filename = askopenfilename(initialdir="images")
    show_image(filename)

def show_image(image_path):
    load = Image.open(image_path)
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img_label.config(image=render)
    img_label.image = render

def preprocess():
    global filename
    global frame
    global faces
    text.delete('1.0', END)
    frame = cv2.imread(filename, 0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    text.insert(END, "Total number of faces detected: " + str(len(faces)))

def detectEmotion():
    global faces
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        
        # Show emotion in uppercase letters with inverted commas
        text.insert(END, f"\nEmotion Detected: \"{label.upper()}\"")
        
        # Show notification and play song in separate threads
        threading.Thread(target=show_notification, args=(label,)).start()
        threading.Thread(target=play_song, args=(label,)).start()
    else:
        messagebox.showinfo("Emotion Prediction Screen", "No face detected in uploaded image")

def show_notification(label):
    # Show a notification for the detected emotion
    messagebox.showinfo("Emotion Detected", f"Emotion Detected: \"{label.upper()}\"")

def play_song(label):
    path = 'songs'
    for r, d, f in os.walk(path):
        for file in f:
            if file.find(label) != -1:
                playsound(os.path.join(path, file))
                return

font = ('times', 20, 'bold')
title = Label(main, text='EMOTION BASED MUSIC RECOMMENDATION SYSTEM')
title.config(bg='brown', fg='white')
title.config(font=font)
title.config(height=3, width=80)
title.place(x=5, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image With Face", command=upload)
upload.place(x=50, y=100)
upload.config(font=font1)

preprocessbutton = Button(main, text="Preprocess & Detect Face in Image", command=preprocess)
preprocessbutton.place(x=50, y=150)
preprocessbutton.config(font=font1)

emotion = Button(main, text="Detect Emotion and Play Song", command=detectEmotion)
emotion.place(x=50, y=200)
emotion.config(font=font1)

# Place img_label between title and buttons, aligned right of buttons and above text widget
img_label = Label(main, bg='white')
img_label.place(x=400, y=100, width=250, height=250)

text = Text(main, height=10, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=400)
text.config(font=font1)

main.config(bg='brown')
main.mainloop()
