# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:56:39 2024

@author: Ahmed
"""

import nest_asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import face_recognition
import os
import io

nest_asyncio.apply()
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# تحميل مجموعة البيانات والعثور على الترميزات
path = r'C:\Users\HP 2021\Desktop\gr_proj\streamlit\dataset'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

@app.post('/recognize')
async def recognize(file: UploadFile = File(...)):
    try:
        image = await file.read()
        img = face_recognition.load_image_file(io.BytesIO(image))
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                return JSONResponse(content={"name": name})

        return JSONResponse(content={"name": "not found"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="192.168.1.62", port=8000)

