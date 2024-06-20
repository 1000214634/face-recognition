import streamlit as st
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import concurrent.futures


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open(r"C:\Users\HP 2021\Desktop\gr_proj\Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
            
            
            
def main():
    st.title("Face Recognition App")
    st.write("Click the button below to start face recognition:")
    
    if st.button("Start Recognition"):
       path = r'C:\Users\HP 2021\Desktop\gr_proj\streamlit\dataset'
       images = []
       classNames = []
       myList = os.listdir(path)
       
       
       for cl in myList:
           curImg = cv2.imread(f'{path}/{cl}')
           images.append(curImg)
           classNames.append(os.path.splitext(cl)[0])

       encodeListKnown = findEncodings(images)
       print('Encoding Complete')
       
       
       cap = cv2.VideoCapture(0)
       frame_resizing = 0.25
       

       
       while True:
           success, img = cap.read()
           imgS = cv2.resize(img, (0, 0), fx=frame_resizing, fy=frame_resizing)
           imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

           facesCurFrame = face_recognition.face_locations(imgS)
           encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
           
           
           
           for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
              matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
              faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
              matchIndex = np.argmin(faceDis)
              if matches[matchIndex]:
                  name = classNames[matchIndex].upper()
                  markAttendance(name)
                  
                  
if __name__ == "__main__":
    main()
       
       
       
       

       
       
       
       
       
       
       
    
    
    

  
      

              
              
              
    
    
    
    
    
    
    