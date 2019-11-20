from flask import Flask, request, render_template
import os
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import os
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def getvalue():
    known_face_encodings = np.load('./testEncodings/known_face_encodings.npy')
    known_face_names = np.load('./testEncodings/known_face_names.npy')
    known_face_encodings = known_face_encodings.tolist()
    known_face_names = known_face_names.tolist()
    url = request.form['name']
    unknown_image = face_recognition.load_image_file(url+".jpg")

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    #draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        A = face_distances
        n = 1
        result = []
        name = []
        for i in range(n):
            mini = np.argmin(A)
            if A[mini]<1.1:
                A[mini]=1.1
                result.append(mini)
        for i in result:
            if matches[i]:
                name.append(known_face_names[i])
    return render_template('results.html', n=name)
