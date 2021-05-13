import os
import imutils
from PIL import Image, ExifTags
from flask import Flask, request, render_template, redirect, url_for, jsonify
from random import randint
import numpy as np
import secrets
from BGRemoval import *
from processing import *
from face_align import *
from orientation import *
from smile import smile_inference
from crop_rotate import *


 
template_path = os.path.join('templates')
static_path = os.path.join('static')

app = Flask(__name__, template_folder=template_path, static_folder=static_path)
dir_path =  os.path.join('temp/')


@app.route('/')
def home():
   return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():

    if 'files' not in request.files:
        return "file input error"

    tempName = dir_path + "temp.jpg"
    
    uploaded_files = request.files.getlist("files")
    file = uploaded_files[0]
    file.save(tempName)
    
    
    width = int(int(request.form['width']) * 3.7795275591)
    height = int(int(request.form['height']) * 3.7795275591)
    

    red = int(request.form['red'])
    green = int(request.form['green'])
    blue = int(request.form['blue'])
    print(red,green,blue,"=========================")
    img_path = dir_path + "temp.jpg"
    
    # Orientation code
    orient = orientation(img_path)
    print(orient,"******************************************")
    img_path = dir_path + "orientation.png"

    image = cv2.imread(img_path)

    # detector = dlib.get_frontal_face_detector()
   
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # rects = detector(gray, 0)
    image = cv2.cvtColor(cv2.imread("temp/BGR.jpg"), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    rects = detector.detect_faces(image)

    RotateFace1(image,rects)

    GetFaceRect1(image, width, height, red,rects)









    smile_check = False
    single_face = False
    red_eye = False
    
    if len(rects) == 1:
        single_face = True
        smile_check = smile_inference(img_path)
        
    
    landmark = face_landmark(image)
    print(smile_check,single_face,"====================================")
    
    if len(rects) ==1:
        if landmark is not None :
            if not RotateFace(img_path): return
            image = face_alignment(img_path)
            cv2.imwrite(dir_path + "align.jpg", image)
            BG = BGRemove()
            img_path = dir_path + "align_prev.jpg"
            # img_path = dir_path + "align.jpg"
            rgb = (red,green,blue)
            image = BG.inference(img_path,rgb)
            cv2.imwrite(dir_path + "BGR.jpg", image)
            crop_img = GetFaceRect(image, width, height, red)
            cv2.imwrite(dir_path + "crop.jpg", crop_img)
        

    elif len(rects)>1:
            BG = BGRemove()
            rgb = (red,green,blue)
            image = BG.inference(img_path,rgb)
            cv2.imwrite(dir_path + "BGR.jpg", image)
    
    

    secret = secrets.token_hex(16)
    secret_public = secrets.token_hex(16)
    file_path = dir_path + "static/upload/" + secret + ".jpg"
    watermarked_path = dir_path + "static/upload/" + secret_public + "_watermarked"+".jpg"
    file_path_4_6 = dir_path + "static/upload/" + secret + "_4x6" + ".jpg"
    
   
    bg_removed_url = "http://api.perfectpassportphotos.com/facecrop/static/upload/" + secret + ".jpg"
    api_response = {
    "bg_removed_url": bg_removed_url,
    "watermarked": "http://api.perfectpassportphotos.com/facecrop/static/upload/" +secret_public+ "_watermarked"+".jpg",
    "photo_dimension_url": "",
    "template_4x6": "http://api.perfectpassportphotos.com/facecrop/static/upload/" +secret+ "_4x6"+".jpg",
    "sentimental_analysis":{
        "smile_check": smile_check,
        "single_face": single_face,
        "red_eye": red_eye
    }
    }

    return api_response

# application = app
if __name__ == "__main__":

    port = 7000
    host = '0.0.0.0'
    app.run(host=host, port=port, threaded=True)
