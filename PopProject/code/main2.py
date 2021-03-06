import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m
from math import sin, cos, radians

WIDTH, HEIGHT = 28, 10 
dim = WIDTH * HEIGHT 
face = cv2.CascadeClassifier("/home/pooja/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml")
settings = {
    'scaleFactor': 1.3, 
    'minNeighbors': 3, 
    'minSize': (50, 50), 
    'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT|cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
}

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]


def show(area): 
    cv.Rectangle(img,(area[0][0],area[0][1]),
                     (area[0][0]+area[0][2],area[0][1]+area[0][3]),
                    (255,0,0),2)
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    cv.ShowImage('Face Detection', img) 
    cv.WaitKey()

def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] 
    return crop
def cropface(x,y,w,h):
    cropface=cf[y: y + h, x: x + w]

def vectorize(filename):
    size = WIDTH, HEIGHT 
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) 
    im_grey = resized_im.convert('L') 
    im_array = np.array(im_grey) 
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array
if __name__ == '__main__':
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    labels = []
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)  
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    
    #x=int(raw_input("Enter the amount of time (in ms) the video has to be captured:"))
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture("vid.mp4")
    
    
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False

    print "\n\n\n\n\npress space to take picture; press ESC to exit"
    now=time.time()
    future=now+17
    smile=0
    nosmile=0
    while True:
        rval, frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        i=1
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if True:
                cv.SaveImage("webcam.jpg", cv.fromarray(frame))
                print x,y,w,h
               
                img = cv.LoadImage("webcam.jpg")
                
                cropping=img[y-30: y + h+50, x-30: x + w+50]
                if(i==1):
                    cv.SaveImage("crop1.jpg", cropping)
                else:
                    cv.SaveImage("crop2.jpg", cropping)
              
                if(i==1):
                    img = cv.LoadImage("crop1.jpg")
                else:
                    img= cv.LoadImage("crop2.jpg")
                mouth = m.findmouth(img)
               
                if mouth != 2: 
                    mouthimg = crop(mouth)
                    if(i==1):
                        cv.SaveImage("webcam-m1.jpg", mouthimg)
                    else:
                        cv.SaveImage("webcam-m2.jpg", mouthimg)
                  
                    if(i==1):
                        result = lr.predict(vectorize('webcam-m1.jpg'))
                    else:
                        result = lr.predict(vectorize('webcam-m2.jpg'))
                    if result == 1:
                        print "face",i, ": You are smiling! :-) "
                        smile=smile+1
                    else:
                        print "face",i, ":You are not smiling :-/ "
                        nosmile=nosmile+1
                else:
                    print "face",i, ":Failed to detect mouth. our face is tilted."
                i=i+1
            cv2.imshow('preview', frame)
        if cv2.waitKey(5) != -1:
            break
        if future < time.time():
            break
       
    print "The smile percentage: ", (smile*100)/(smile+nosmile)
    print "The nosmile percentage: ", (nosmile*100)/(smile+nosmile)
    cv2.destroyWindow("preview")
