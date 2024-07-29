# Creating database 
# It captures images and stores them in datasets 
# folder under the folder name of sub_data 
import cv2
import numpy as np

import os


haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file) 

# All the faces data will be present this folder 
datasets = "Enter your dataset folder location here"



# These are sub data sets of folder, 
# All datasets can be trained for any person by using his folder name
# change the name here 
sub_data = input("Enter the name of the person to train his data:")
webcam = cv2.VideoCapture(0)

path = os.path.join(datasets, sub_data) 
if not os.path.isdir(path): 
	os.mkdir(path)
# Create a face recognizer object


# defining the size of images 
(width, height) = (130, 100)	 

#'0' is used for webcam, 

 

# The program loops until it has 30 images of the face. 
count = 1
while count < 120: 
	(_, im) = webcam.read() 
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 4)
	
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
	count += 1
	
	cv2.imshow('OpenCV', im) 
	key = cv2.waitKey(10) & 0xff
	if key==ord('q'): 
		break
