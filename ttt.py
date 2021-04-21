# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
from BGRemove import *
import numpy as np
from processing import *
from face_align import *
from smile import smile_inference

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)

dir_path = 'temp'
# flip filename 'vertical' or 'horizontal'
@app.route("/flip", methods=["POST"])
def flip():
    red = int(request.form['red'])
    green = int(request.form['green'])
    blue = int(request.form['blue'])
    print(red,green,blue)
    w = 55
    h = 55
    width = w * 3.7795275591
    height = h * 3.7795275591

    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    img = Image.open(destination)
    print(img.size,type(img))
    img = np.array(img) 
    # Face Detection module
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    print((rects))

    if len(rects) ==1:
        image = face_alignment(img)
        cv2.imwrite(dir_path + "align.jpg", image)
        BG = BGRemove()
        img_path = dir_path + "align.jpg"
        rgb = (red,green,blue)
        image = BG.inference(img_path,rgb)
        cv2.imwrite(dir_path + "BGR.jpg", image)
        image = GetFaceRect(image, width, height, red)
        cv2.imwrite(dir_path + "crop.jpg", image)
        

    elif len(rects)>1:
        BG = BGRemove()
        rgb = (red,green,blue)
        image = BG.inference(img_path,rgb)
        cv2.imwrite(dir_path + "BGR.jpg", image)

    img = Image.fromarray((image).astype(np.uint8))

    # save and return image
    destination = "/".join([target, 'temp.png'])
    print(destination)
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image('temp.png')

# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    print(send_from_directory("static/images", filename),filename)
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run()

