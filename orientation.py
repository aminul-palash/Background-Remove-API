from mtcnn import MTCNN
import imutils
import cv2

def orientation(img_path):
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.imread(img_path)
    detector = MTCNN()
    p = detector.detect_faces(img)
    
    if len(p)==0:
        rotated = imutils.rotate(img, 90)
        detector = MTCNN()
        p1 = detector.detect_faces(rotated)
        rotated2 = imutils.rotate(rotated, 180)
        detector = MTCNN()
        p2 = detector.detect_faces(rotated2)
        # cv2.imwrite("test1.png",rotated)
        # cv2.imwrite("test2.png",rotated2)
       
        if len(p1)==0 and len(p2)==0:
            rotated4 = imutils.rotate(img, 180)
            detector = MTCNN()
            p4= detector.detect_faces(rotated4)
            if len(p4)==1:
                # cv2.imwrite("temp/orientation.png",rotated4)
                return rotated4,True
            elif len(p4)>1:
                # cv2.imwrite("temp/orientation.png",rotated4)
                return rotated4,False
            else:
                return None
        elif len(p1)>1:
            # cv2.imwrite("temp/orientation.png",rotated4)
            return rotated4,False
        elif len(p2)>1:
            # cv2.imwrite("temp/orientation.png",rotated4)
            return rotated4,False
        elif len(p1)==1 and len(p2)==0:
            # cv2.imwrite("temp/orientation.png",rotated)
            return rotated,True
        elif len(p1)==0 and len(p2)==1:
            # cv2.imwrite("temp/orientation.png",rotated2)
            return rotated2,True
            
        elif len(p1)==1 and len(p2)==1:
            if p1[0]['confidence'] > p2[0]['confidence']:
                # cv2.imwrite("temp/orientation.png",rotated)
                return rotated,True
            else:
                # cv2.imwrite("temp/orientation.png",rotated2)
                return rotated2,True
        else:
            return None
    
    elif len(p)>1:
        return img,False

    else:
        
        rotated3 = imutils.rotate(img, 180)
        detector = MTCNN()
        q = detector.detect_faces(rotated3)
       
        if (len(q)==1 or len(p)==1) and len(q)!=0:
            
            if p[0]['confidence'] > q[0]['confidence']:
                # cv2.imwrite("temp/orientation.png",img)
                return img,True
            else:
                # cv2.imwrite("temp/orientation.png",rotated3)
                return rotated3,True
        elif len(q)==0 and len(p)==1:
            # cv2.imwrite("temp/orientation.png",img)
            return img,True
        else:
            # cv2.imwrite("temp/orientation.png",img)
            return img,False
        
    

if __name__=="__main__":
    img_path = 'family.jpg'
    output = orientation(img_path)
    if output==None:
        print("No human faces detected")
    else:
        img,b = output
        if b==False:
            print("More than one face detected")
            cv2.imwrite('filename.jpg', img)


        
        