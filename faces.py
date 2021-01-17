import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickle",'rb') as f:
    rev_labels = pickle.load(f)
    labels = {v:k for k,v in rev_labels.items()}

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
#i=1
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #cv2.imwrite(f'.\\images\\temp\\{i}.png', frame)
    #i=i+1
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y+h+25), font, 0.5, color, stroke, cv2.LINE_AA)
        color = (255, 0, 0) # BGR
        stroke = 2  # Thickness of rectangle line
        x_init = x
        y_init = y
        x_final = x + w
        y_final = y + h
        cv2.rectangle(frame, (x_init, y_init), (x_final, y_final), color, stroke) 
    cv2.imshow('frame', frame)
    # out.write(frame)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()