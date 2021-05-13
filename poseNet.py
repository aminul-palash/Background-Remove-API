import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 

import math


def parse_output(heatmap_data,offset_data, threshold):
  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''
  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):
      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1
  return pose_kps


def new_coordinates_after_resize_img(original_size, new_size, original_coordinate):
  original_size = np.array(original_size)
  new_size = np.array(new_size)
  original_coordinate = np.array(original_coordinate)
  xy = original_coordinate/(original_size/new_size)
  x, y = int(xy[0]), int(xy[1])
  return (x, y)


def poseNet(input):
    model_path = "pretrained_model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    
    height,width = input.shape[0],input.shape[1]
    test_img = cv2.resize(input,(input_shape[1],input_shape[2]))
    h,w = test_img.shape[0],test_img.shape[1]
    template_input = np.expand_dims(test_img.copy(), axis=0)
    floating_model = input_details[0]['dtype'] == np.float32

    if floating_model:
        template_input = (np.float32(template_input) - 127.5) / 127.5
    
    interpreter.set_tensor(input_details[0]['index'], template_input)
    interpreter.invoke()
    template_output_data = interpreter.get_tensor(output_details[0]['index'])
    template_offset_data = interpreter.get_tensor(output_details[1]['index'])
    template_heatmaps = np.squeeze(template_output_data)
    template_offsets = np.squeeze(template_offset_data)
    template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
    template_show = np.array(template_show*255,np.uint8)
    template_kps = parse_output(template_heatmaps,template_offsets,0.3)
    
    left_neck_x = template_kps[6][1]
    left_neck_y = template_kps[6][0]

    right_neck_x = template_kps[5][1]
    right_neck_y = template_kps[5][0]
    
    # print((left_neck_y,left_neck_x))
    # cv2.circle(test_img,(left_neck_x,left_neck_y),6,(243,56,32),-1)
    # cv2.circle(test_img,(right_neck_x,right_neck_y),6,(243,56,32),-1)
    # cv2.imwrite('neck.jpg', test_img)

    orig_left_neck_x,orig_left_neck_y = new_coordinates_after_resize_img((h,w), (width,height), (left_neck_x, left_neck_y)) # just modify this line
    orig_right_neck_x,orig_right_neck_y = new_coordinates_after_resize_img((h,w), (width,height), (right_neck_x, right_neck_y)) # just modify this line
    
    orig_w = abs(orig_left_neck_x-orig_right_neck_x)

    cv2.circle(input,(orig_left_neck_x,orig_left_neck_y),10,(243,56,32),-1)
    cv2.circle(input,(orig_right_neck_x,orig_right_neck_y),10,(243,56,32),-1)
    cv2.imwrite('orig.jpg', input)
   
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()
    rects = detector.detect_faces(input)
    eye = rects[0]['keypoints']['left_eye']
    dist = min(abs(orig_left_neck_y-eye[1]),orig_left_neck_y)
    
    y = orig_left_neck_y
    x = max(orig_left_neck_x-40,40)

    h = min(dist+int(dist//1.2),orig_left_neck_y)
    print(h,"=============")
    neck_dist = min(orig_right_neck_x+40,orig_right_neck_x+abs(orig_right_neck_x+width))
    w = abs(neck_dist-x)
   
    crop_img = input[y-h+20:y, x:x+w]
    cv2.imwrite("cropped.jpg", crop_img)
    return crop_img
    



if __name__=="__main__":
  template_path = "12.jpg"
  input = cv2.imread(template_path)
  poseNet(input)
