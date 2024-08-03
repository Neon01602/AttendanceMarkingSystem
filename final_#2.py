import cv2
import numpy 
import mysql.connector
import os
from datetime import date
cnx = mysql.connector.connect(
    user='root',
    password='12345',
    host='localhost',
    database='attendance'
)
haar_file = r'C:\Users\Hp\Downloads\haarcascade_frontalface_default.xml'
cursor = cnx.cursor()
datasets = r'D:\project\datasets'
face_cascade = cv2.CascadeClassifier(haar_file) 
print('Recognizing Face Please Be in sufficient Lights...') 
(images, labels, names, id) = ([], [], {}, 0) 
for (subdirs, dirs, files) in os.walk(datasets): 
	for subdir in dirs: 
		names[id] = subdir 
		subjectpath = os.path.join(datasets, subdir) 
		for filename in os.listdir(subjectpath): 
			path = subjectpath + '/' + filename 
			label = id
			images.append(cv2.imread(path, 0)) 
			labels.append(int(label)) 
		id += 1
(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]  
 
model = cv2.face.LBPHFaceRecognizer_create()  
model.train(images, labels) 
cap = cv2.VideoCapture(0)
marked = {}
while True:
    (_, im) = cap.read()
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        if prediction[1]<500:
                cv2.putText(im, '% s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                if names[prediction[0]] not in marked:
                        cursor.execute("INSERT INTO attendance (student_id, attendance_date, attendance_status, student_name) VALUES (%s, %s, %s, %s)", (prediction[0], date.today(), 'Present', names[prediction[0]]))
                        marked[names[prediction[0]]] = True
                        cnx.commit()
                        print(f"Attendance marked for {names[prediction[0]]}")
        else:
            cv2.putText(im, 'Not Recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        
                
    cv2.imshow('Face Recognition', im)
        
    cursor.execute("SELECT * FROM attendance")
    attendance_data = cursor.fetchall()
    attendance_image = numpy.zeros((500, 500, 3), numpy.uint8)
    y = 50


    key = cv2.waitKey(10) & 0xff
    if key == ord('q'):
        break
for row in attendance_data:
        temp=row[4]
        print("Student ID:", row[0])
        print("Name:", row[4])
        print("Date:", row[2])
        print("Status:", row[3])

        
cap.release()
cv2.destroyAllWindows()
cursor.close()
cnx.close()
