import cv2
import mediapipe as mp
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

model = load_model('./signlanguage/models/model3.h5')
model.summary()

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape

analysisframe = ''

while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            diff = (y_max-y_min) - (x_max-x_min)
            if(diff > 0):
                x_min = x_min - int(diff / 2)
                x_max = x_max + int(diff / 2)
            else:
                y_min = y_min + int(diff / 2)
                y_max = y_max - int(diff / 2)
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            def inFrame():
                if(y_min > 0 and x_min > 0 and y_max < 480 and x_max < 640):
                    return True
                else:
                    return False
            if(inFrame()):
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                analysisframe = frame
                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe,(28,28))
                # img = cv2.normalize(analysisframe, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                img = img_to_array(analysisframe)
                img = img / 255.0
                # plt.imshow(img, cmap='gray')
                # plt.waitforbuttonpress() 
                img = np.expand_dims(img, axis=0)
                prediction = model.predict(img)
                # print(prediction)
                text_idx=np.argmax(prediction)
                # print("Predicted character: " , labels[text_idx] , " with condidence of ", int(prediction[0][text_idx]*100),"%")
                cv2.putText(frame, labels[text_idx]+" "+str(int(prediction[0][text_idx]*100))+"%", (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            # mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()