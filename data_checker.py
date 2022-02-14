from asyncio.windows_events import NULL

import os

from PIL import Image

import numpy as np

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR,"Faces_Train")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
counter = 0
for root, dirs, files in os.walk(image_dir):

    for file in files:

        if file.endswith("png") or file.endswith("jpg"):

            path=os.path.join(root, file)

            pil_image = Image.open(path).convert("L")

            image_array = np.array(pil_image,"uint8")
            gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
            
            if faces != []:
                counter = counter + 1
                
print(counter)
