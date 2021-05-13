import os
import argparse
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import helper
from PIL import Image

model_path = 'pretrained_model/BGRNet.onnx'

class BGRemove():
    # define hyper-parameters
    
    def __init__(self):
        self.session = onnxruntime.InferenceSession(model_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def get_scale_factor(self,im_h, im_w, ref_size):
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h

        return x_scale_factor, y_scale_factor
    
        
    def file_load(self, image_path):
        
        im = cv2.imread(image_path)
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # unify image channels to 3
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]
        return im
    
    def pre_process(self, im):
        
        # normalize values to scale it between -1 to 1
        im = (im - 127.5) / 127.5   

        im_h, im_w, im_c = im.shape
        ref_size = 512
        x, y = self.get_scale_factor(im_h, im_w, ref_size) 

        # resize image
        im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

        # prepare input shape
        im = np.transpose(im)
        im = np.swapaxes(im, 1, 2)
        im = np.expand_dims(im, axis = 0).astype('float32')
        return im,im_h, im_w
    
    def combined_display(self,image, matte,rgb):
        # calculate display resolution
        w, h = image.width, image.height
        rw, rh = 800, int(h * 800 / (3 * w))

        # obtain predicted foreground
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        
        # foreground = image * matte + np.full(image.shape, rgb) * (1 - matte)
        foreground = image * matte + np.full(image.shape, 255.0) * (1 - matte)
        
        foreground = np.float32(foreground)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
        return foreground
       
    
    def inference(self,image_path,rgb=None):
        im = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        # im = self.file_load(image_path)
        input_img = Image.fromarray(im)
        
        im,im_h, im_w = self.pre_process(im)
        
        result = self.session.run([self.output_name], {self.input_name: im})
        matte = (np.squeeze(result[0]) * 255).astype('uint8')
        matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)
       
        matte = Image.fromarray(matte)
        return self.combined_display(input_img, matte,rgb)

if __name__ == '__main__':
    
    BG = BGRemove()
    image_path = 'images/1.jpg'
    rgb = (255.0,0.0,0.0)
    img = cv2.imread(image_path)
    output = BG.inference(img,rgb)
    cv2.imwrite('test.png',output)
    print(output.shape,type(output))
    
    
