import os
import cv2
import imutils
import numpy as np
import pickle
from threading import Thread
from tkinter import *
from tkinter import filedialog, messagebox
from gtts import gTTS
from playsound import playsound
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
from keras.models import Sequential, model_from_json

# Global variables
bg = None
playcount = 0
names = ['C', 'Thumbs Down', 'Fist', 'I', 'Ok', 'Palm', 'Thumbs up']  # Example names list, adjust as needed

# Function to delete all .mp3 files in 'play' directory
def deleteDirectory():
    filelist = [f for f in os.listdir('play') if f.endswith(".mp3")]
    for f in filelist:
        os.remove(os.path.join('play', f))

# Function to play a gesture as speech using gTTS and playsound
def play(playcount, gesture):
    class PlayThread(Thread):
        def __init__(self, playcount, gesture):
            Thread.__init__(self)
            self.gesture = gesture
            self.playcount = playcount

        def run(self):
            # Convert text to speech and save as mp3
            tts = gTTS(text=self.gesture, lang='en', slow=False)
            tts.save(f"play/{self.playcount}.mp3")
            # Play the saved mp3 file
            playsound(f"play/{self.playcount}.mp3")

    # Start a new thread to play the gesture
    newthread = PlayThread(playcount, gesture)
    newthread.start()

# Function to remove background using MOG2 method
def remove_background(frame):
    global bg
    if bg is None:
        bg = cv2.createBackgroundSubtractorMOG2(0, 50)
    fgmask = bg.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Function to upload dataset directory
def uploadDataset():
    global filename
    global labels
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, f"{filename} loaded\n\n")

# Function to train the CNN model
def trainCNN():
    global classifier
    text.delete('1.0', END)
    # Assuming X_train and Y_train are already loaded or generated
    X_train = np.load('model1/X.txt.npy')  # Example path, adjust as needed
    Y_train = np.load('model1/Y.txt.npy')  # Example path, adjust as needed
    text.insert(END, f"CNN is training on total images: {len(X_train)}\n")
    
    if os.path.exists('model1/model.json'):
        # Load existing model
        with open('model1/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model1/model_weights.h5")
        classifier._make_predict_function()
        print(classifier.summary())

        # Load training history
        with open('model1/history.pckl', 'rb') as f:
            data = pickle.load(f)
        acc = data['accuracy']
        accuracy = acc[-1] * 100  # Assuming accuracy is the last value in history
        text.insert(END, f"CNN Hand Gesture Training Model Prediction Accuracy: {accuracy:.2f}%\n")
    else:
        # Create new model
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(256, activation='relu'))
        classifier.add(Dense(len(names), activation='softmax'))
        print(classifier.summary())

        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)

        classifier.save_weights('model1/model_weights.h5')
        model_json = classifier.to_json()
        with open("model1/model.json", "w") as json_file:
            json_file.write(model_json)
        with open('model1/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

        # Display final accuracy after training
        with open('model1/history.pckl', 'rb') as f:
            data = pickle.load(f)
        acc = data['accuracy']
        accuracy = acc[-1] * 100  # Assuming accuracy is the last value in history
        text.insert(END, f"CNN Hand Gesture Training Model Prediction Accuracy: {accuracy:.2f}%\n")

# Function to initialize background averaging
def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

# Function to segment hand from background
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented

# Function to predict gesture from webcam feed
def webcamPredict():
    global playcount
    oldresult = 'none'
    count = 0
    fgbg2 = cv2.createBackgroundSubtractorKNN()
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 325, 690
    num_frames = 0

    while True:
        grabbed, frame = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (41, 41), 0)

        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255), 2)

                roi = frame[top:bottom, right:left]
                roi = fgbg2.apply(roi)
                cv2.imwrite("test.jpg", roi)
                img = cv2.imread("test.jpg")
                img = cv2.resize(img, (64, 64))
                img = img.reshape(1, 64, 64, 3)
                img = np.array(img, dtype='float32')
                img /= 255

                predict = classifier.predict(img)
                value = np.amax(predict)
                cl = np.argmax(predict)
                result = names[np.argmax(predict)]

                if value >= 0.99:
                    cv2.putText(clone, f'Gesture Recognized as: {result}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    if oldresult != result:
                        play(playcount, result)
                    oldresult = result
                    playcount += 1
                else:
                    cv2.putText(clone, '', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.imshow("Video Feed", clone)
                keypress = cv2.waitKey(1) & 0xFF
                if keypress == ord("q"):
                    break

            num_frames += 1

    camera.release()
    cv2.destroyAllWindows()

# Tkinter GUI setup
main = Tk()
main.title("Hand Gesture Recognition and Voice Conversation using CNN")
main.geometry("1200x600")

font = ('times', 16, 'bold')
title = Label(main, text='Hand Gesture Recognition and Voice Conversation using CNN', anchor=W, justify=CENTER)
title.config(bg='green', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Hand Gesture Dataset", command=uploadDataset)
upload.place(x=50, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel
