import cv2
import numpy as np
import mysql.connector
import os
from datetime import date
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(background="#333333")  # Dark grey background

        # Create frames
        self.frame1 = tk.Frame(self.root, bg="#333333")
        self.frame1.pack(fill="x")
        self.frame2 = tk.Frame(self.root, bg="#333333")
        self.frame2.pack(fill="x")
        self.frame3 = tk.Frame(self.root, bg="#333333")
        self.frame3.pack(fill="x")
        self.frame4 = tk.Frame(self.root, bg="#333333")
        self.frame4.pack(fill="x")

        # Create labels and buttons
        self.label1 = tk.Label(self.frame1, text="Face Recognition System", font=("Arial", 24, "bold"), bg="#333333", fg="white")
        self.label1.pack(pady=20)
        self.button_frame = tk.Frame(self.frame2, bg="#333333")
        self.button_frame.pack(pady=10)
        self.button1 = tk.Button(self.button_frame, text="Start Recognition", command=self.start_recognition, font=("Arial", 18, "bold"), bg="#CCCCCC", fg="black", borderwidth=2, relief="ridge")
        self.button1.pack(side=tk.LEFT, padx=10)
        self.button2 = tk.Button(self.button_frame, text="View Attendance", command=self.view_attendance, font=("Arial", 18, "bold"), bg="#CCCCCC", fg="black", borderwidth=2, relief="ridge")
        self.button2.pack(side=tk.LEFT, padx=10)
        self.button3 = tk.Button(self.button_frame, text="Exit", command=self.root.destroy, font=("Arial", 18, "bold"), bg="#CCCCCC", fg="black", borderwidth=2, relief="ridge")
        self.button3.pack(side=tk.LEFT, padx=10)

        # Create text box
        self.text_box = tk.Text(self.frame3, width=40, height=10, font=("Arial", 12), bg="#FFFFFF", fg="black", borderwidth=2, relief="ridge")
        self.text_box.pack(side=tk.LEFT, padx=10)

        # Create video label
        self.video_label = tk.Label(self.frame3, bg="#FFFFFF", borderwidth=2, relief="ridge")
        self.video_label.pack(side=tk.LEFT, padx=10)

        # Initialize variables
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.threshold = 100
        self.haar_file = r'C:\Users\Hp\Downloads\haarcascade_frontalface_default.xml'
        self.cnx = mysql.connector.connect(
            user='root',
            password='12345',
            host='localhost',
            database='attendance'
        )
        self.cursor = self.cnx.cursor()
        self.datasets = r'D:\project\datasets'
        self.face_cascade = cv2.CascadeClassifier(self.haar_file)
        self.images, self.labels, self.names, self.id = [], [], {}, 0
        self.load_datasets()

    def load_datasets(self):
        for (subdirs, dirs, files) in os.walk(self.datasets):
            for subdir in dirs:
                self.names[self.id] = subdir
                subjectpath = os.path.join(self.datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    label = self.id
                    self.images.append(cv2.imread(path, 0))
                    self.labels.append(int(label))
                self.id += 1
        (self.width, self.height) = (130, 100)
        (self.images, self.labels) = [np.array(lis) for lis in [self.images, self.labels]]
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(self.images, self.labels)

    def start_recognition(self):
            self.cap = cv2.VideoCapture(0)
            self.marked = set()

            def update_video():
                (_, im) = self.cap.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face_resize = cv2.resize(face, (self.width, self.height))
                                       
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    prediction = self.model.predict(face_resize)
                    confidence = prediction[1]
                    if confidence < self.threshold:
                        cv2.putText(im, '% s - %.0f' %(self.names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                        if prediction[0] in self.marked:
                            self.TP += 1
                        else:
                            self.FN += 1
                        if prediction[0] not in self.marked:
                            self.marked.add(prediction[0])
                            self.cursor.execute("INSERT INTO attendance (student_id, attendance_date, attendance_status, student_name) VALUES (%s, %s, %s, %s)", (prediction[0], date.today(), 'Present', self.names[prediction[0]]))
                            self.cnx.commit()
                            print(f"Attendance marked for {self.names[prediction[0]]}")
                            self.FP += 1
                    else:
                        cv2.putText(im, 'Not Recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                        self.TN += 1

                cv2_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_im)
                img = ImageTk.PhotoImage(img)
                self.video_label.config(image=img)
                self.video_label.image = img
                self.root.after(1, update_video)  # Update GUI every 1ms

            update_video()  # Start the update loop

    def view_attendance(self):
        self.cursor.execute("SELECT * FROM attendance")
        attendance_data = self.cursor.fetchall()
        self.text_box.delete(1.0, tk.END)
        for row in attendance_data:
            self.text_box.insert(tk.END, f"Student ID: {row[0]}\nName: {row[4]}\nDate: {row[2]}\nStatus: {row[3]}\n\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionSystem(root)
    root.mainloop()
