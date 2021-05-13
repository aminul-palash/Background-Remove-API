import cv2
import imutils
import math

 
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def AlignFace(image,rects):
   
    success = False
   
    
    
    eye_line = [rects[0]['keypoints']['right_eye'][0]-rects[0]['keypoints']['left_eye'][0] , rects[0]['keypoints']['right_eye'][1]-rects[0]['keypoints']['left_eye'][1] ]

    # eye_line = [landmark[1] - landmark[0], landmark[6] - landmark[5]]
    roll_angle = math.atan2(eye_line[1], eye_line[0])
    aligned_face = rotate(image, math.degrees(roll_angle))
    
    return aligned_face


def GetFaceRect(image, wid, hei, red,rects):
#     image = imutils.resize(image, width=500)

    # landmark = face_landmark(image)
    # print(landmark[2], landmark[8] ,landmark[7])
    # bottom_point = [int(landmark[2]), int(2 * landmark[8] - landmark[7])]
    bottom_point = [int(rects[0]['keypoints']['nose'][0]), int(2* rects[0]['keypoints']['mouth_left'][1] - rects[0]['keypoints']['nose'][1])]

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

    h = int((bottom_point[1] - top_point[1]) * 100 / 72)
    w = int(h * wid / hei)
    y = int(top_point[1] - (bottom_point[1] - top_point[1]) * 8 / 55)
    x = int(bottom_point[0] - w / 2)
    
    if y < 0: y = 0
    if x < 0: x = 0
    if y + h > img_h: h = img_h - y
    if x + w > img_w: w = img_w - x
    print(x,y,w,h)
    face_img = image[y:y + h, x:x + w]
    cv2.imwrite("temp/getrect_latest.jpg", face_img)
    return face_img


if __name__ == '__main__':
    
    from mtcnn.mtcnn import MTCNN
    image_path = 'images/1.jpg'
    
    image = cv2.imread(image_path)
    detector = MTCNN()
    rects = detector.detect_faces(image)
    aligned_face = AlignFace(image,rects)
    cv2.imwrite("hello.jpg",aligned_face)
