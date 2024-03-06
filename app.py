# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:04:40 2024

@author: Ahmed
"""

import streamlit as st
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def main():
    st.title("Face Recognition App")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, flags=3)

        # Convert image to RGB format
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find faces and their encodings
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Load and encode known faces from the dataset
        path = r'C:\Users\HP 2021\Desktop\gr_proj\streamlit\dataset'
        images = []
        classNames = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

        encodeListKnown = findEncodings(images)

        # Process each detected face
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                st.success(f"Match Found: {name}")
            else:
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Not Found", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                st.error("No Match Found")

        # Display the processed image
        st.image(img, channels="BGR")


if __name__ == "__main__":
    main()



