# app.py
from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Face detection
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (138, 43, 226), 3)

    result_path = 'result_' + filename
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_path), img)

    return render_template('index.html', result_image=result_path)

if __name__ == '__main__':
    app.run(debug=True)
