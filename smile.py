import keras
import time
from keras.models import model_from_json
model = model_from_json(open('pretrained_model/smile.json').read())
model.load_weights('pretrained_model/smile.h5')

import numpy as np

def print_indicator(data, model, class_names, bar_width=50):
    probabilities = model.predict(np.array([data]))[0]

    left_count = int(probabilities[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    print(class_names[0], left_side + '###' + right_side, class_names[1])
class_names = ['Neutral', 'Smiling']

import cv2

classifier = cv2.CascadeClassifier('pretrained_model/haarcascade_frontalface_default.xml')



def smile_inference(img_path):
    frame = cv2.imread(img_path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r = 320.0 /gray.shape[1]
    dim = (320, int(gray.shape[0] * r))
    resized = cv2 .resize(gray, dim, interpolation = cv2.INTER_AREA)
    faces = classifier.detectMultiScale(resized, 1.3, 5)

    if len(faces) != 0:
        (x,y,w,h) = faces[0]
        
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
            
        detected_face = resized[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.resize(detected_face, (32, 32))

        data = detected_face[:, :, np.newaxis]
        data = data.astype(np.float) / 255
        probabilities = model.predict(np.array([data]))[0]
        # A = [False,True]
        # return A[np.argmax(probabilities)]
        bar_width = 100
        left_count = int(probabilities[1] * bar_width)
        
        right_count = bar_width - left_count
        print(left_count,right_count)
        left_side = '-' * left_count
        right_side = '-' * right_count
        print(class_names[0], left_side + '###' + right_side, class_names[1])
        return probabilities

if __name__ == '__main__':
    image_path = 'temp/temp.jpg'
    print(smile_inference(image_path))
   
