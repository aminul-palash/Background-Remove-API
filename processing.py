import cv2
import imutils
import math
import tensorflow.compat.v1 as tf

import detect_face

minsize = 50
detection_threshold = [0.7, 0.8, 0.9]
factor = 0.709
gpu_memory_fraction = 1.0


def face_landmark(src):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
        )
        sess = tf.Session(
            config=tf.ConfigProto(
                gpu_options=gpu_options,
                log_device_placement=False,
            )
        )
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess)

    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    face_boxes, landmarks = detect_face.detect_face(
        img,
        minsize,
        pnet,
        rnet,
        onet,
        detection_threshold,
        factor,
    )

    if len(face_boxes) != 1:
        return None
    else:
        land = landmarks[:, 0]
        return land

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def RotateFace(filePath):
    image = cv2.imread(filePath)
    # image = imutils.resize(image, width=2000)
    success = False
    landmark = face_landmark(image)
  
    if landmark.any != None:
        eye_line = [landmark[1] - landmark[0], landmark[6] - landmark[5]]
        roll_angle = math.atan2(eye_line[1], eye_line[0])
        rotated = rotate(image, math.degrees(roll_angle))
        cv2.imwrite("temp/align_prev.jpg", rotated)
        success = True
    return success


def GetFaceRect(image, wid, hei, red):
    # image = imutils.resize(image, width=500)

    landmark = face_landmark(image)
    print(landmark[2], landmark[8] ,landmark[7])
    bottom_point = [int(landmark[2]), int(2 * landmark[8] - landmark[7])]

    istop = 0
    img_h, img_w, _ = image.shape
    for k in range(img_h):
        for j in range(img_w):
            if image[k, j, 2] != red:
                istop = 1
                break;
        if istop == 1:
            break;

    top_point = [j, k]

    h = int((bottom_point[1] - top_point[1]) * 100 / 62)
    w = int(h * wid / hei)
    y = int(top_point[1] - (bottom_point[1] - top_point[1]) * 8 / 62)
    x = int(bottom_point[0] - w / 2)
    if y < 0: y = 0
    if x < 0: x = 0
    if y + h > img_h: h = img_h - y
    if x + w > img_w: w = img_w - x

    face_img = image[y:y + h, x:x + w]
    return face_img

