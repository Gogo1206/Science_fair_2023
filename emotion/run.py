import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import img_to_array

model = load_model("./emotion/models/model.h5")

img = Image.open("./emotion/tmp/img.jpg")
img = img.convert('L')
np.expand_dims(img, axis=-1)
print(img.size)
img = img.resize((48,48))
y = img_to_array(img)
plt.imshow(y,cmap='gray')
plt.show()
plt.waitforbuttonpress
y = np.expand_dims(y, axis=0)
pred = model.predict(y)
text_idx=np.argmax(pred)
text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print(text_list[text_idx])


# cap = cv2.VideoCapture(0)
# import copy

# while True:
    
#     ret, frame = cap.read()
#     img = copy.deepcopy(frame)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         fc = gray[y:y+h, x:x+w]
        
#         roi = cv2.resize(fc, (48,48))
#         pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
#         text_idx=np.argmax(pred)
#         text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#         if text_idx == 0:
#             text= text_list[0]
#         if text_idx == 1:
#             text= text_list[1]
#         elif text_idx == 2:
#             text= text_list[2]
#         elif text_idx == 3:
#             text= text_list[3]
#         elif text_idx == 4:
#             text= text_list[4]
#         elif text_idx == 5:
#             text= text_list[5]
#         elif text_idx == 6:
#             text= text_list[6]
#         cv2.putText(img, text, (x, y-5),
#            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
#         img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            
    
#     cv2.imshow("frame", img)
#     key = cv2.waitKey(1) & 0xFF
#     if key== ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()