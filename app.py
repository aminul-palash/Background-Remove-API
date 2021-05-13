import os
import imutils
from flask import Flask, request, render_template, redirect, url_for, jsonify
from random import randint
import numpy as np
import secrets
from BGRemoval import *
from orientation import *
from smile import smile_inference
from crop_rotate import *
from poseNet import poseNet


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

    img_path = dir_path + "temp.jpg"
    
    uploaded_files = request.files.getlist("files")
    file = uploaded_files[0]
    file.save(img_path)
    
    
    width = int(int(request.form['width']) * 3.7795275591)
    height = int(int(request.form['height']) * 3.7795275591)
    

    red = int(request.form['red'])
    green = int(request.form['green'])
    blue = int(request.form['blue'])
    
    single_face = False
    smile_check = False
    
    output = orientation(img_path)
    if output==None:
        print("No human faces detected")
    else:
        img,b = output
        if b==False:
            print("More than one face detected")
            
        else:
           
            single_face = True
            smile_status,value = smile_inference(img)
            BG = BGRemove()
         
            rgb = (red,green,blue)
            image = BG.inference(img,rgb)

          
            detector = MTCNN()
            rects = detector.detect_faces(image)
            aligned_face = AlignFace(image,rects)
           
          
            crop_img = poseNet(aligned_face)
           
            cv2.imwrite(dir_path + "crop_img.jpg", crop_img)
    

    
    data = {}


    if single_face == False:
        data["single_face"]=False
    else:
        data["single_face"]=False
        data["smile"]=[smile_status,value]
    

    return data


if __name__ == "__main__":

    port = 7000
    host = '0.0.0.0'
    app.run(host=host, port=port, threaded=True)