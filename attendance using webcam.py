import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#import pywhatkit
path = 'Imgattendance'
images = []
cn = []
usn = []
classNames = []
time = []
phoneno = []

mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)


def extractNames():
    with open('studentList.csv', 'r+') as f:
        list = f.readlines()
        for line in list:
            entry = line.split(',')
            classNames.append(entry[0])
            usn.append(entry[1])
            #phoneno.append(entry[2])


extractNames()
print(usn)
print(phoneno)
print(classNames)


def encoding(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList


def markattendance(name, usn):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dstring = now.strftime("%H:%M:%S")
            time = dstring.split(':')
            f.writelines(f'\n{name},{dstring},{usn}')
            #msg='Your ward '+name+' USN ' +usn+' has attended 1st period'
            # pywhatkit.sendwhatmsg(phoneno,msg,int(time[0]),int(time[1])+2)


encodelist = encoding(images)
cap=cv2.VideoCapture(0)


while True:
    sucess,img=cap.read();
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encode = face_recognition.face_encodings(imgs, facesCurFrame)

    for encodeface, faceloc in zip(encode, facesCurFrame):
        match = face_recognition.compare_faces(encodelist, encodeface)
        facedis = face_recognition.face_distance(encodelist, encodeface)
        matchIndex = np.argmin(facedis)
        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            usnAttended = usn[matchIndex]
            #student_phoneno = phoneno[matchIndex]

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markattendance(name, usnAttended)

    cv2.imshow('Webcam', img)
    cv2.waitKey(0)
