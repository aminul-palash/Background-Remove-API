import keras
import time
import cv2
from keras.models import model_from_json
from mtcnn import MTCNN
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


def smile_inference(img_path):
    # frame = cv2.imread(img_path)
    frame = img_path

    img_mt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    p = detector.detect_faces(img_mt)
   
    x,y,w,h = p[0]['box'][0], p[0]['box'][1], p[0]['box'][2], p[0]['box'][3]
    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
    
    gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
    r = 320.0 /gray.shape[1]
    dim = (320, int(gray.shape[0] * r))
    resized = cv2 .resize(gray, dim, interpolation = cv2.INTER_AREA)
    detected_face = cv2.resize(resized, (32, 32))
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

    if left_count >= right_count:
        return True,left_count
    else:
        return False,left_count


if __name__ == '__main__':
    image_path = 'temp/temp.jpg'
    print(smile_inference(image_path))
   
